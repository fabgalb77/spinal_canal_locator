#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for dual-headed spinal canal localization.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import json
from tqdm import tqdm
import random
import sys
from typing import Dict, List, Tuple, Optional, Union, Any

from data.preprocessing import create_processed_dataset, setup_logger
from data.DualHeadCanalDataset import DualHeadCanalDataset, DualHeadCanalDataModule
from models.DualHeadCanalModel import DualHeadCanalModel


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dual-headed spinal canal localization model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Data directory (overrides config)"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0,
        help="GPU ID to use (-1 for CPU)"
    )
    
    parser.add_argument(
        "--force_preprocess", 
        action="store_true",
        help="Force preprocessing of data"
    )
    
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        help="Include negative examples (slices without certain levels)"
    )
    
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=1.,
        help="Ratio of negative examples to include"
    )
    
    parser.add_argument(
        "--classification_weight",
        type=float,
        default=5.0,
        help="Weight for classification loss"
    )
    
    parser.add_argument(
        "--localization_weight",
        type=float,
        default=1.0,
        help="Weight for localization loss"
    )
    
    return parser.parse_args()


class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance.
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, inputs, targets):
        """Forward pass."""
        # Apply sigmoid to get probabilities
        inputs_prob = torch.sigmoid(inputs)
        inputs_prob = torch.clamp(inputs_prob, self.eps, 1.0 - self.eps)
        
        # Calculate BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Apply gamma focusing parameter
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def dice_loss(inputs, targets, smooth=1.0):
    """
    Dice loss for segmentation tasks.
    
    Args:
        inputs: Model outputs (logits)
        targets: Ground truth
        smooth: Smoothing factor
        
    Returns:
        Dice loss
    """
    # Apply sigmoid to get probabilities
    inputs = torch.sigmoid(inputs)
    
    # Flatten
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # Calculate Dice score
    intersection = (inputs * targets).sum()
    dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    # Return Dice loss
    return 1 - dice_score


class LocalizationLoss(nn.Module):
    """
    Combined loss function for localization task (BCE + Dice + Focal).
    """
    
    def __init__(
        self, 
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Alpha parameter for Focal loss
            pos_weight: Weight for positive examples
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize losses
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight)
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
            
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )
    
    def forward(self, inputs, targets):
        """Forward pass."""
        bce = self.bce_loss(inputs, targets) * self.bce_weight
        dice = dice_loss(inputs, targets) * self.dice_weight
        focal = self.focal_loss(inputs, targets) * self.focal_weight
        
        return bce + dice + focal


class DualTaskLoss(nn.Module):
    """
    Combined loss function for both localization and classification tasks.
    """
    
    def __init__(
        self, 
        localization_loss,
        classification_weight: float = 1.0,
        localization_weight: float = 1.0
    ):
        """
        Initialize dual task loss.
        
        Args:
            localization_loss: Loss function for the localization task
            classification_weight: Weight for classification loss
            localization_weight: Weight for localization loss
        """
        super().__init__()
        self.localization_loss = localization_loss
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
    
    def forward(self, outputs, targets):
        """
        Forward pass.
        
        Args:
            outputs: Tuple of (heatmap, classification)
            targets: Tuple of (heatmap, level_present)
            
        Returns:
            Combined loss and individual losses
        """
        # Unpack outputs and targets
        pred_heatmap, pred_classification = outputs
        target_heatmap, target_classification = targets
        
        # Calculate localization loss
        loc_loss = self.localization_loss(pred_heatmap, target_heatmap)
        
        # Calculate classification loss
        cls_loss = self.classification_loss(
            pred_classification, 
            target_classification
        )
        
        # Combine losses
        combined_loss = (
            self.localization_weight * loc_loss + 
            self.classification_weight * cls_loss
        )
        
        return combined_loss, loc_loss, cls_loss


def train_one_epoch(
    model, 
    dataloader, 
    optimizer, 
    loss_fn, 
    device, 
    epoch, 
    logger
):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch
        logger: Logger
        
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    running_loc_loss = 0.0
    running_cls_loss = 0.0
    
    # Track level-specific losses
    level_losses = {i: {'loc': 0.0, 'cls': 0.0, 'combined': 0.0} for i in range(5)}
    level_counts = {i: 0 for i in range(5)}
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # Get data
        images = batch["image"].to(device)
        heatmaps = batch["heatmap"].unsqueeze(1).to(device)  # Add channel dimension [B, 1, H, W]
        level_indices = batch["level_idx"].to(device)
        level_present = batch["level_present"].to(device)  # [B, 1]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get batch size
        batch_size = images.shape[0]
        
        # Initialize batch loss
        batch_loss = 0.0
        batch_loc_loss = 0.0
        batch_cls_loss = 0.0
        
        # Process each sample individually
        for i in range(batch_size):
            img = images[i:i+1]  # Keep batch dimension
            target_heatmap = heatmaps[i:i+1]  # Keep batch dimension
            level_idx = level_indices[i].item()
            target_present = level_present[i:i+1]
            
            # Forward pass for this specific level
            pred_heatmap, pred_present = model(img, level_idx=level_idx)
            
            # Calculate loss
            loss, loc_loss, cls_loss = loss_fn(
                (pred_heatmap, pred_present),
                (target_heatmap, target_present)
            )
            
            # Accumulate loss
            batch_loss += loss
            batch_loc_loss += loc_loss
            batch_cls_loss += cls_loss
            
            # Track level-specific loss
            level_losses[level_idx]['loc'] += loc_loss.item()
            level_losses[level_idx]['cls'] += cls_loss.item()
            level_losses[level_idx]['combined'] += loss.item()
            level_counts[level_idx] += 1
        
        # Normalize batch loss
        batch_loss = batch_loss / batch_size
        
        # Backward pass
        batch_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update running loss
        running_loss += batch_loss.item()
        running_loc_loss += batch_loc_loss.item() / batch_size
        running_cls_loss += batch_cls_loss.item() / batch_size
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": batch_loss.item(),
            "loc_loss": batch_loc_loss.item() / batch_size,
            "cls_loss": batch_cls_loss.item() / batch_size
        })
    
    # Calculate average losses
    avg_loss = running_loss / len(dataloader)
    avg_loc_loss = running_loc_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    
    # Calculate average level-specific losses
    avg_level_losses = {}
    for level_idx, losses in level_losses.items():
        count = level_counts[level_idx]
        if count > 0:
            avg_level_losses[level_idx] = {
                'loc': losses['loc'] / count,
                'cls': losses['cls'] / count,
                'combined': losses['combined'] / count
            }
        else:
            avg_level_losses[level_idx] = {'loc': 0.0, 'cls': 0.0, 'combined': 0.0}
    
    # Log losses
    logger.info(
        f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f} "
        f"(Loc: {avg_loc_loss:.6f}, Cls: {avg_cls_loss:.6f})"
    )
    
    for level_idx, losses in avg_level_losses.items():
        level_name = model.get_level_name(level_idx)
        logger.info(
            f"  Level {level_name}: Combined: {losses['combined']:.6f}, "
            f"Loc: {losses['loc']:.6f}, Cls: {losses['cls']:.6f} "
            f"({level_counts[level_idx]} samples)"
        )
    
    return avg_loss, avg_loc_loss, avg_cls_loss, avg_level_losses


def validate(
    model, 
    dataloader, 
    loss_fn, 
    device, 
    epoch, 
    logger,
    visualize=False,
    save_dir=None
):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch
        logger: Logger
        visualize: Whether to visualize predictions
        save_dir: Directory to save visualizations
        
    Returns:
        Average validation loss and metrics
    """
    model.eval()
    running_loss = 0.0
    running_loc_loss = 0.0
    running_cls_loss = 0.0
    
    # Track level-specific losses and metrics
    level_losses = {i: {'loc': 0.0, 'cls': 0.0, 'combined': 0.0} for i in range(5)}
    level_distances = {i: [] for i in range(5)}
    level_cls_accuracies = {i: [] for i in range(5)}
    level_counts = {i: 0 for i in range(5)}
    
    # Track classification metrics
    all_pred_cls = []
    all_true_cls = []
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    # Images to visualize (one per level if possible)
    vis_images = {i: None for i in range(5)}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch["image"].to(device)
            heatmaps = batch["heatmap"].unsqueeze(1).to(device)  # Add channel dimension [B, 1, H, W]
            level_indices = batch["level_idx"].to(device)
            level_present = batch["level_present"].to(device)  # [B, 1]
            
            # Get scaled coordinates (already in target image space)
            scaled_coords = batch["scaled_coordinates"]
            scaled_x = scaled_coords[0].numpy()
            scaled_y = scaled_coords[1].numpy()
            
            # Get batch size
            batch_size = images.shape[0]
            
            # Initialize batch loss
            batch_loss = 0.0
            batch_loc_loss = 0.0
            batch_cls_loss = 0.0
            
            # Process each sample individually
            for i in range(batch_size):
                img = images[i:i+1]  # Keep batch dimension
                target_heatmap = heatmaps[i:i+1]  # Keep batch dimension
                level_idx = level_indices[i].item()
                target_present = level_present[i:i+1]
                
                # Store image for visualization if this level hasn't been visualized yet
                if visualize and vis_images[level_idx] is None:
                    vis_images[level_idx] = {
                        'image': img.cpu(),
                        'target_heatmap': target_heatmap.cpu(),
                        'target_present': target_present.cpu(),
                        'level_idx': level_idx,
                        'level_name': batch["level"][i],
                        'coords': (scaled_x[i], scaled_y[i])
                    }
                
                # Forward pass for this specific level
                pred_heatmap, pred_present = model(img, level_idx=level_idx)
                
                # Calculate loss
                loss, loc_loss, cls_loss = loss_fn(
                    (pred_heatmap, pred_present),
                    (target_heatmap, target_present)
                )
                
                # Accumulate loss
                batch_loss += loss.item()
                batch_loc_loss += loc_loss.item()
                batch_cls_loss += cls_loss.item()
                
                # Track level-specific loss
                level_losses[level_idx]['loc'] += loc_loss.item()
                level_losses[level_idx]['cls'] += cls_loss.item()
                level_losses[level_idx]['combined'] += loss.item()
                level_counts[level_idx] += 1
                
                # Calculate classification accuracy
                pred_cls = torch.sigmoid(pred_present).cpu().numpy()
                true_cls = target_present.cpu().numpy()
                
                all_pred_cls.append(pred_cls[0, 0])
                all_true_cls.append(true_cls[0, 0])
                
                # Calculate classification accuracy
                pred_cls_label = (pred_cls > 0.5).astype(float)
                cls_correct = (pred_cls_label == true_cls).astype(float)
                level_cls_accuracies[level_idx].append(cls_correct[0, 0])
                
                # For positive examples, calculate distance
                if true_cls[0, 0] > 0.5:
                    # Extract predicted coordinates
                    output_np = pred_heatmap[0, 0].cpu().numpy()
                    prob_map = 1 / (1 + np.exp(-output_np))  # sigmoid
                    pred_y, pred_x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                    
                    # Get ground truth coordinates
                    gt_x, gt_y = scaled_x[i], scaled_y[i]
                    
                    # Calculate distance
                    distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                    level_distances[level_idx].append(distance)
            
            # Update running loss
            running_loss += batch_loss / batch_size
            running_loc_loss += batch_loc_loss / batch_size
            running_cls_loss += batch_cls_loss / batch_size
        
        # Calculate average loss
        avg_loss = running_loss / len(dataloader)
        avg_loc_loss = running_loc_loss / len(dataloader)
        avg_cls_loss = running_cls_loss / len(dataloader)
        
        # Calculate average level-specific losses and metrics
        avg_level_losses = {}
        avg_level_distances = {}
        median_level_distances = {}
        avg_level_cls_accuracies = {}
        
        for level_idx in range(5):
            count = level_counts[level_idx]
            if count > 0:
                # Losses
                avg_level_losses[level_idx] = {
                    'loc': level_losses[level_idx]['loc'] / count,
                    'cls': level_losses[level_idx]['cls'] / count,
                    'combined': level_losses[level_idx]['combined'] / count
                }
                
                # Classification accuracy
                accuracies = level_cls_accuracies[level_idx]
                if accuracies:
                    avg_level_cls_accuracies[level_idx] = np.mean(accuracies)
                else:
                    avg_level_cls_accuracies[level_idx] = 0.0
                
                # Distance metrics (only for positive examples)
                distances = level_distances[level_idx]
                if distances:
                    avg_level_distances[level_idx] = np.mean(distances)
                    median_level_distances[level_idx] = np.median(distances)
                else:
                    avg_level_distances[level_idx] = 0.0
                    median_level_distances[level_idx] = 0.0
            else:
                avg_level_losses[level_idx] = {'loc': 0.0, 'cls': 0.0, 'combined': 0.0}
                avg_level_cls_accuracies[level_idx] = 0.0
                avg_level_distances[level_idx] = 0.0
                median_level_distances[level_idx] = 0.0
        
        # Calculate global average distance (only for positive examples)
        all_distances = [d for dists in level_distances.values() for d in dists]
        avg_distance = np.mean(all_distances) if all_distances else 0.0
        median_distance = np.median(all_distances) if all_distances else 0.0
        
        # Calculate global classification accuracy
        all_pred_cls_labels = (np.array(all_pred_cls) > 0.5).astype(float)
        all_true_cls = np.array(all_true_cls)
        cls_accuracy = np.mean((all_pred_cls_labels == all_true_cls).astype(float))
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {avg_loss:.6f} "
            f"(Loc: {avg_loc_loss:.6f}, Cls: {avg_cls_loss:.6f}), "
            f"Cls Acc: {cls_accuracy:.4f}, "
            f"Avg Distance: {avg_distance:.2f} px, "
            f"Median Distance: {median_distance:.2f} px"
        )
        
        for level_idx in range(5):
            level_name = model.get_level_name(level_idx)
            if level_counts[level_idx] > 0:
                logger.info(
                    f"  Level {level_name}: "
                    f"Loss: {avg_level_losses[level_idx]['combined']:.6f} "
                    f"(Loc: {avg_level_losses[level_idx]['loc']:.6f}, "
                    f"Cls: {avg_level_losses[level_idx]['cls']:.6f}), "
                    f"Cls Acc: {avg_level_cls_accuracies[level_idx]:.4f}, "
                    f"Avg Dist: {avg_level_distances[level_idx]:.2f} px, "
                    f"Median Dist: {median_level_distances[level_idx]:.2f} px "
                    f"({level_counts[level_idx]} samples)"
                )
        
        # Visualize predictions
        if visualize and save_dir is not None:
            # Create save directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Create a figure with one row per level
            num_levels_to_viz = sum(1 for v in vis_images.values() if v is not None)
            
            if num_levels_to_viz > 0:
                fig, axes = plt.subplots(num_levels_to_viz, 4, figsize=(20, 5 * num_levels_to_viz))
                
                # Handle case where only one level is visualized
                if num_levels_to_viz == 1:
                    axes = axes.reshape(1, 4)
                
                row = 0
                
                # Visualize each level
                for level_idx in range(5):
                    data = vis_images[level_idx]
                    if data is None:
                        continue
                    
                    # Get data
                    img = data['image'][0].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 0.5) + 0.5  # Denormalize
                    
                    target_heatmap = data['target_heatmap'][0, 0].cpu().numpy()
                    target_present = data['target_present'][0, 0].item()
                    level_name = data['level_name']
                    
                    # Only show coordinates for positive examples
                    if target_present > 0.5:
                        gt_x, gt_y = data['coords']
                    else:
                        gt_x, gt_y = -1, -1
                    
                    # Run model to get prediction
                    with torch.no_grad():
                        pred_heatmap, pred_present = model(
                            data['image'].to(device), 
                            level_idx=level_idx
                        )
                        pred_heatmap_np = pred_heatmap[0, 0].cpu().numpy()
                        prob_map = 1 / (1 + np.exp(-pred_heatmap_np))  # sigmoid
                        
                        # Get classification prediction
                        pred_present_prob = torch.sigmoid(pred_present).item()
                        pred_present_label = int(pred_present_prob > 0.5)
                    
                    # Get predicted coordinates
                    pred_y, pred_x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                    
                    # Calculate distance if positive example
                    if target_present > 0.5:
                        distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                        distance_text = f"Distance: {distance:.2f} px"
                    else:
                        distance_text = "N/A (Negative Example)"
                    
                    # Plot image with points
                    axes[row, 0].imshow(img[:, :, 0], cmap='gray')
                    
                    # Only show points for positive examples
                    if target_present > 0.5:
                        axes[row, 0].scatter(gt_x, gt_y, c='r', marker='x', s=100, label='Ground Truth')
                        axes[row, 0].scatter(pred_x, pred_y, c='b', marker='o', s=100, label='Prediction')
                        axes[row, 0].legend()
                    
                    # Set title based on true/pred present
                    title = f"Level {level_name}\n"
                    title += f"True: {'Present' if target_present > 0.5 else 'Not Present'}, "
                    title += f"Pred: {'Present' if pred_present_label else 'Not Present'} ({pred_present_prob:.2f})"
                    axes[row, 0].set_title(title)
                    
                    # Plot ground truth heatmap
                    axes[row, 1].imshow(target_heatmap, cmap='hot')
                    axes[row, 1].set_title("Ground Truth Heatmap")
                    
                    # Plot predicted heatmap
                    axes[row, 2].imshow(prob_map, cmap='hot')
                    axes[row, 2].set_title(f"Predicted Heatmap\n{distance_text}")
                    
                    # Plot overlay of predicted heatmap on image
                    axes[row, 3].imshow(img[:, :, 0], cmap='gray')
                    axes[row, 3].imshow(prob_map, cmap='hot', alpha=0.5)
                    axes[row, 3].set_title("Heatmap Overlay")
                    
                    row += 1
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"val_epoch_{epoch+1}.png"))
                plt.close()
        
        # Return metrics
        return (
            avg_loss, avg_loc_loss, avg_cls_loss, 
            avg_distance, median_distance, cls_accuracy,
            avg_level_losses, avg_level_distances, median_level_distances, avg_level_cls_accuracies
        )


def visualize_multiple_samples(
    model, 
    train_dataloader, 
    val_dataloader, 
    device, 
    output_dir,
    epoch,
    num_samples=3
):
    """
    Visualize both training and validation samples.
    
    Args:
        model: Model to use for predictions
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        device: Device to use
        output_dir: Directory to save visualizations
        epoch: Current epoch
        num_samples: Number of samples to visualize per dataset
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Visualize training samples
    _visualize_dataset_samples(
        model=model,
        dataloader=train_dataloader,
        device=device,
        output_path=os.path.join(vis_dir, f"train_samples.png"),
        title=f"Training Samples - Epoch {epoch+1}",
        num_samples=num_samples
    )
    
    # Visualize validation samples
    _visualize_dataset_samples(
        model=model,
        dataloader=val_dataloader,
        device=device,
        output_path=os.path.join(vis_dir, f"val_samples.png"),
        title=f"Validation Samples - Epoch {epoch+1}",
        num_samples=num_samples
    )


def _visualize_dataset_samples(
    model, 
    dataloader, 
    device, 
    output_path, 
    title, 
    num_samples=3
):
    """
    Visualize samples from a dataset.
    
    Args:
        model: Model to use for predictions
        dataloader: Dataloader for the dataset
        device: Device to use
        output_path: Path to save the output image
        title: Title for the plot
        num_samples: Number of samples to visualize
    """
    # Get a batch of data
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    
    # Create a figure with one row per sample
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    
    # Process each sample
    for i in range(min(num_samples, len(batch['image']))):
        # Get data for this sample
        image = batch['image'][i:i+1].to(device)
        heatmap = batch['heatmap'][i:i+1].to(device)
        level_idx = batch['level_idx'][i].item()
        level_name = batch['level'][i]
        level_present = batch['level_present'][i].item()
        
        # Get scaled coordinates
        scaled_x, scaled_y = batch['scaled_coordinates'][0][i].item(), batch['scaled_coordinates'][1][i].item()
        
        # Forward pass
        with torch.no_grad():
            pred_heatmap, pred_present = model(image, level_idx=level_idx)
            
            # Get predictions for all levels too
            all_heatmaps, all_classifications = model.forward_all_levels(image)
        
        # Denormalize image
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5) + 0.5  # Denormalize
        
        # Get target heatmap
        target_heatmap = heatmap[0].cpu().numpy()
        
        # Get predicted heatmap
        pred_heatmap_np = pred_heatmap[0, 0].cpu().numpy()
        pred_prob = 1 / (1 + np.exp(-pred_heatmap_np))  # Apply sigmoid
        
        # Get classification prediction
        pred_present_prob = torch.sigmoid(pred_present).item()
        pred_present_label = int(pred_present_prob > 0.5)
        
        # Find predicted peak coordinates
        pred_y, pred_x = np.unravel_index(pred_prob.argmax(), pred_prob.shape)
        
        # Calculate distance if positive example
        if level_present > 0.5:
            distance = np.sqrt((pred_x - scaled_x) ** 2 + (pred_y - scaled_y) ** 2)
            distance_text = f"Distance: {distance:.2f} px"
        else:
            distance_text = "N/A (Negative Example)"
        
        # Plot image with points
        axes[i, 0].imshow(img_np[:, :, 0], cmap='gray')
        
        # Only show points for positive examples
        if level_present > 0.5:
            axes[i, 0].scatter(scaled_x, scaled_y, c='r', marker='x', s=100, label='Ground Truth')
            axes[i, 0].scatter(pred_x, pred_y, c='b', marker='o', s=100, label='Prediction')
            axes[i, 0].legend()
        
        # Set title based on true/pred present
        title_text = f"Level {level_name}\n"
        title_text += f"True: {'Present' if level_present > 0.5 else 'Not Present'}, "
        title_text += f"Pred: {'Present' if pred_present_label else 'Not Present'} ({pred_present_prob:.2f})"
        axes[i, 0].set_title(title_text)
        
        # Plot ground truth heatmap
        axes[i, 1].imshow(target_heatmap, cmap='hot')
        axes[i, 1].set_title("Ground Truth Heatmap")
        
        # Plot predicted heatmap
        axes[i, 2].imshow(pred_prob, cmap='hot')
        axes[i, 2].set_title(f"Predicted Heatmap\n{distance_text}")
        
        # Plot overlay of predicted heatmap on image
        axes[i, 3].imshow(img_np[:, :, 0], cmap='gray')
        axes[i, 3].imshow(pred_prob, cmap='hot', alpha=0.5)
        axes[i, 3].set_title("Heatmap Overlay")
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close(fig)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.output_dir is not None:
        config['training']['output_dir'] = args.output_dir
    
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['training']['output_dir'], f"dual_head_canal_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("train_dual_head", os.path.join(output_dir, "train.log"))
    
    # Set random seed
    set_seed(config['data']['seed'])
    
    # Create processed dataset
    logger.info("Preprocessing data...")
    processed_data = create_processed_dataset(
        data_dir=config['data']['data_dir'],
        series_csv=config['data']['series_csv'],
        coordinates_csv=config['data']['coordinates_csv'],
        output_dir=os.path.join(output_dir, "processed_data"),
        force_reprocess=args.force_preprocess,
        logger=logger
    )
    
    # Create data module
    logger.info("Creating data module...")
    data_module = DualHeadCanalDataModule(
        data_dir=config['data']['data_dir'],
        canal_slices_file=processed_data['canal_slices'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        target_size=tuple(config['data']['target_size']),
        split_ratio=config['data']['split_ratio'],
        seed=config['data']['seed'],
        include_negatives=args.include_negatives,
        negative_ratio=args.negative_ratio
    )
    
    # Set up datasets
    data_module.setup()
    
    # Set device
    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("Using CPU")
    
    # Create model
    logger.info("Creating dual-headed model...")
    model = DualHeadCanalModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        in_channels=config['model']['in_channels'],
        dropout_rate=config['model']['dropout_rate']
    )
    model = model.to(device)
    
    # Create loss functions
    logger.info("Creating loss functions...")
    
    # Localization loss (BCE + Dice + Focal)
    localization_loss = LocalizationLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        pos_weight=config['training']['loss'].get('pos_weight', None)
    )
    
    # Combined dual task loss
    loss_fn = DualTaskLoss(
        localization_loss=localization_loss,
        classification_weight=args.classification_weight,
        localization_weight=args.localization_weight
    )
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create scheduler
    if config['training']['scheduler']['name'] == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            verbose=True
        )
    else:
        scheduler = None
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Save config
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # Train
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs...")
    
    # Track best model
    best_val_distance = float('inf')
    best_val_loss = float('inf')
    best_val_cls_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=data_module.train_dataloader(),
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        train_loss, train_loc_loss, train_cls_loss, train_level_losses = train_metrics
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=data_module.val_dataloader(),
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            logger=logger,
            visualize=True,
            save_dir=os.path.join(output_dir, "visualizations")
        )
        
        (
            val_loss, val_loc_loss, val_cls_loss, 
            val_distance, val_median_distance, val_cls_accuracy,
            val_level_losses, val_level_distances, val_level_median_distances, val_level_cls_accuracies
        ) = val_metrics
        
        # Visualize multiple samples
        visualize_multiple_samples(
            model=model,
            train_dataloader=data_module.train_dataloader(),
            val_dataloader=data_module.val_dataloader(),
            device=device,
            output_dir=output_dir,
            epoch=epoch,
            num_samples=3
        )
        
        # Update TensorBoard
        writer.add_scalar("Loss/Train/Combined", train_loss, epoch)
        writer.add_scalar("Loss/Train/Localization", train_loc_loss, epoch)
        writer.add_scalar("Loss/Train/Classification", train_cls_loss, epoch)
        
        writer.add_scalar("Loss/Val/Combined", val_loss, epoch)
        writer.add_scalar("Loss/Val/Localization", val_loc_loss, epoch)
        writer.add_scalar("Loss/Val/Classification", val_cls_loss, epoch)
        
        writer.add_scalar("Metrics/Val/Distance", val_distance, epoch)
        writer.add_scalar("Metrics/Val/MedianDistance", val_median_distance, epoch)
        writer.add_scalar("Metrics/Val/ClassificationAccuracy", val_cls_accuracy, epoch)
        
        # Add level-specific metrics
        for level_idx in range(5):
            level_name = model.get_level_name(level_idx)
            
            # Training losses
            if level_idx in train_level_losses:
                writer.add_scalar(
                    f"Loss/Train/Level_{level_name}/Combined", 
                    train_level_losses[level_idx]['combined'], 
                    epoch
                )
                writer.add_scalar(
                    f"Loss/Train/Level_{level_name}/Localization", 
                    train_level_losses[level_idx]['loc'], 
                    epoch
                )
                writer.add_scalar(
                    f"Loss/Train/Level_{level_name}/Classification", 
                    train_level_losses[level_idx]['cls'], 
                    epoch
                )
            
            # Validation metrics
            if level_idx in val_level_losses:
                writer.add_scalar(
                    f"Loss/Val/Level_{level_name}/Combined", 
                    val_level_losses[level_idx]['combined'], 
                    epoch
                )
                writer.add_scalar(
                    f"Loss/Val/Level_{level_name}/Localization", 
                    val_level_losses[level_idx]['loc'], 
                    epoch
                )
                writer.add_scalar(
                    f"Loss/Val/Level_{level_name}/Classification", 
                    val_level_losses[level_idx]['cls'], 
                    epoch
                )
            
            if level_idx in val_level_distances:
                writer.add_scalar(
                    f"Metrics/Val/Level_{level_name}/Distance", 
                    val_level_distances[level_idx], 
                    epoch
                )
                writer.add_scalar(
                    f"Metrics/Val/Level_{level_name}/MedianDistance", 
                    val_level_median_distances[level_idx], 
                    epoch
                )
            
            if level_idx in val_level_cls_accuracies:
                writer.add_scalar(
                    f"Metrics/Val/Level_{level_name}/ClassificationAccuracy", 
                    val_level_cls_accuracies[level_idx], 
                    epoch
                )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_distance": val_distance,
            "val_cls_accuracy": val_cls_accuracy
        }
        
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # Check if this is the best model (based on a combined metric)
        # We want high accuracy and low distance
        combined_metric = (1.0 - val_cls_accuracy) + (val_distance / 100.0)  # Normalize distance
        best_combined_metric = (1.0 - best_val_cls_acc) + (best_val_distance / 100.0)
        
        if combined_metric < best_combined_metric:
            best_val_distance = val_distance
            best_val_loss = val_loss
            best_val_cls_acc = val_cls_accuracy
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                checkpoint,
                os.path.join(output_dir, "best_model.pth")
            )
            logger.info(
                f"New best model saved with combined metric: {combined_metric:.4f} "
                f"(Cls Acc: {val_cls_accuracy:.4f}, Dist: {val_distance:.2f} px)"
            )
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(
        checkpoint,
        os.path.join(output_dir, "final_model.pth")
    )
    
    # Log best results
    logger.info(
        f"Best model at epoch {best_epoch+1} with "
        f"Cls Acc: {best_val_cls_acc:.4f}, Distance: {best_val_distance:.2f} px"
    )
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    import torch.nn.functional as F
    main()