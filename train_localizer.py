#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the spinal canal localization model.
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
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import torch.nn.functional as F

from data.preprocessing import create_processed_dataset, setup_logger
from data.CanalLocalizationDataset import CanalLocalizationDataset, CanalLocalizationDataModule
from models.SpinalCanalLocalizationModel import SpinalCanalLocalizationModel


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
    parser = argparse.ArgumentParser(description="Train spinal canal localization model")
    
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
        "--bce_weight",
        type=float,
        default=1.0,
        help="Weight for BCE loss"
    )
    
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Weight for Dice loss"
    )
    
    parser.add_argument(
        "--focal_weight",
        type=float,
        default=0.5,
        help="Weight for Focal loss"
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


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, logger):
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
        Average training loss and metrics
    """
    model.train()
    running_loss = 0.0
    
    # Track level-specific losses and distances
    level_losses = {i: 0.0 for i in range(5)}
    level_distances = {i: [] for i in range(5)}
    level_counts = {i: 0 for i in range(5)}
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # Get data
        images = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)
        level_indices = batch["level_idx"].to(device)
        
        # Get scaled coordinates
        scaled_coords = batch["scaled_coordinates"]
        scaled_x = scaled_coords[0].numpy()
        scaled_y = scaled_coords[1].numpy()
        
        # Get batch size
        batch_size = images.shape[0]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Initialize batch loss
        batch_loss = 0.0
        
        # Process each sample individually
        for i in range(batch_size):
            img = images[i:i+1]  # Keep batch dimension
            target_heatmap = heatmaps[i:i+1]
            level_idx = level_indices[i].item()
            
            # Forward pass for this specific level
            pred_heatmap = model(img, level_idx=level_idx)
            
            # Calculate loss
            loss = loss_fn(pred_heatmap, target_heatmap)
            
            # Accumulate loss
            batch_loss += loss
            
            # Track level-specific loss
            level_losses[level_idx] += loss.item()
            level_counts[level_idx] += 1
            
            # Calculate distance between predicted and ground truth peak
            with torch.no_grad():
                # Extract predicted coordinates
                pred_heatmap_np = pred_heatmap[0, 0].cpu().numpy()
                prob_map = 1 / (1 + np.exp(-pred_heatmap_np))  # sigmoid
                pred_y, pred_x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                
                # Get ground truth coordinates
                gt_x, gt_y = scaled_x[i], scaled_y[i]
                
                # Calculate distance
                distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                level_distances[level_idx].append(distance)
        
        # Normalize batch loss
        batch_loss = batch_loss / batch_size
        
        # Backward pass
        batch_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update running loss
        running_loss += batch_loss.item()
        
        # Update progress bar with the latest distance
        recent_distances = [d for dists in level_distances.values() for d in dists[-5:] if d]
        avg_recent_distance = np.mean(recent_distances) if recent_distances else 0
        
        progress_bar.set_postfix({
            "loss": batch_loss.item(),
            "avg_dist": avg_recent_distance
        })
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    
    # Calculate average distances
    avg_level_distances = {}
    median_level_distances = {}
    
    for level_idx, distances in level_distances.items():
        if distances:
            avg_level_distances[level_idx] = np.mean(distances)
            median_level_distances[level_idx] = np.median(distances)
        else:
            avg_level_distances[level_idx] = 0.0
            median_level_distances[level_idx] = 0.0
    
    # Calculate global average distance
    all_distances = [d for dists in level_distances.values() for d in dists]
    avg_distance = np.mean(all_distances) if all_distances else 0.0
    median_distance = np.median(all_distances) if all_distances else 0.0
    
    # Log metrics
    logger.info(
        f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f}, "
        f"Avg Distance: {avg_distance:.2f} px, "
        f"Median Distance: {median_distance:.2f} px"
    )
    
    # Log level-specific metrics
    for level_idx in range(5):
        level_name = model.get_level_name(level_idx)
        count = level_counts[level_idx]
        
        if count > 0:
            avg_level_loss = level_losses[level_idx] / count
            
            logger.info(
                f"  Level {level_name}: Loss: {avg_level_loss:.6f}, "
                f"Avg Dist: {avg_level_distances[level_idx]:.2f} px, "
                f"Median Dist: {median_level_distances[level_idx]:.2f} px "
                f"({count} samples)"
            )
    
    return {
        'loss': avg_loss,
        'avg_distance': avg_distance,
        'median_distance': median_distance,
        'level_distances': avg_level_distances,
        'level_median_distances': median_level_distances
    }


def validate(model, dataloader, loss_fn, device, epoch, logger, visualize=False, save_dir=None):
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
        Validation metrics
    """
    model.eval()
    running_loss = 0.0
    
    # Track level-specific losses and distances
    level_losses = {i: 0.0 for i in range(5)}
    level_distances = {i: [] for i in range(5)}
    level_counts = {i: 0 for i in range(5)}
    
    # Track samples for visualization
    vis_samples = {i: None for i in range(5)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]"):
            # Get data
            images = batch["image"].to(device)
            heatmaps = batch["heatmap"].to(device)
            level_indices = batch["level_idx"].to(device)
            
            # Get scaled coordinates
            scaled_coords = batch["scaled_coordinates"]
            scaled_x = scaled_coords[0].numpy()
            scaled_y = scaled_coords[1].numpy()
            
            # Get batch size
            batch_size = images.shape[0]
            
            # Process each sample individually
            for i in range(batch_size):
                img = images[i:i+1]  # Keep batch dimension
                target_heatmap = heatmaps[i:i+1]
                level_idx = level_indices[i].item()
                
                # Forward pass for this specific level
                pred_heatmap = model(img, level_idx=level_idx)
                
                # Calculate loss
                loss = loss_fn(pred_heatmap, target_heatmap)
                
                # Update running loss
                running_loss += loss.item()
                
                # Track level-specific loss
                level_losses[level_idx] += loss.item()
                level_counts[level_idx] += 1
                
                # Calculate distance between predicted and ground truth peak
                # Extract predicted coordinates
                pred_heatmap_np = pred_heatmap[0, 0].cpu().numpy()
                prob_map = 1 / (1 + np.exp(-pred_heatmap_np))  # sigmoid
                pred_y, pred_x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                
                # Get ground truth coordinates
                gt_x, gt_y = scaled_x[i], scaled_y[i]
                
                # Calculate distance
                distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                level_distances[level_idx].append(distance)
                
                # Store sample for visualization if it's the best so far (has lowest distance)
                if visualize and (vis_samples[level_idx] is None or distance < vis_samples[level_idx]['distance']):
                    vis_samples[level_idx] = {
                        'image': img.cpu(),
                        'target_heatmap': target_heatmap.cpu(),
                        'pred_heatmap': pred_heatmap.cpu(),
                        'pred_prob': prob_map,
                        'gt_coords': (gt_x, gt_y),
                        'pred_coords': (pred_x, pred_y),
                        'distance': distance,
                        'level_idx': level_idx,
                        'level_name': batch["level"][i]
                    }
        
        # Calculate average loss
        avg_loss = running_loss / len(dataloader.dataset)
        
        # Calculate average distances
        avg_level_distances = {}
        median_level_distances = {}
        
        for level_idx, distances in level_distances.items():
            if distances:
                avg_level_distances[level_idx] = np.mean(distances)
                median_level_distances[level_idx] = np.median(distances)
            else:
                avg_level_distances[level_idx] = 0.0
                median_level_distances[level_idx] = 0.0
        
        # Calculate global average distance
        all_distances = [d for dists in level_distances.values() for d in dists]
        avg_distance = np.mean(all_distances) if all_distances else 0.0
        median_distance = np.median(all_distances) if all_distances else 0.0
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {avg_loss:.6f}, "
            f"Avg Distance: {avg_distance:.2f} px, "
            f"Median Distance: {median_distance:.2f} px"
        )
        
        # Log level-specific metrics
        for level_idx in range(5):
            level_name = model.get_level_name(level_idx)
            count = level_counts[level_idx]
            
            if count > 0:
                avg_level_loss = level_losses[level_idx] / count
                
                logger.info(
                    f"  Level {level_name}: Loss: {avg_level_loss:.6f}, "
                    f"Avg Dist: {avg_level_distances[level_idx]:.2f} px, "
                    f"Median Dist: {median_level_distances[level_idx]:.2f} px "
                    f"({count} samples)"
                )
        
        # Visualize predictions
        if visualize and save_dir:
            visualize_predictions(
                vis_samples=vis_samples,
                save_path=os.path.join(save_dir, f"val_viz_epoch_{epoch+1}.png"),
                epoch=epoch
            )
        
        return {
            'loss': avg_loss,
            'avg_distance': avg_distance,
            'median_distance': median_distance,
            'level_distances': avg_level_distances,
            'level_median_distances': median_level_distances
        }


def visualize_predictions(vis_samples, save_path, epoch):
    """
    Visualize model predictions.
    
    Args:
        vis_samples: Dictionary of visualization samples
        save_path: Path to save visualization
        epoch: Current epoch
    """
    # Count how many samples we have
    valid_samples = [v for v in vis_samples.values() if v is not None]
    
    if not valid_samples:
        return
    
    # Create figure with one row per level
    fig, axes = plt.subplots(len(valid_samples), 3, figsize=(15, 5 * len(valid_samples)))
    
    # Handle case with just one sample
    if len(valid_samples) == 1:
        axes = axes.reshape(1, 3)
    
    # Plot samples
    row = 0
    
    for level_idx in range(5):
        sample = vis_samples[level_idx]
        
        if sample is None:
            continue
        
        # Get data
        img = sample['image'][0].cpu().numpy().transpose(1, 2, 0)
        img = (img * 0.5) + 0.5  # Denormalize
        
        target_heatmap = sample['target_heatmap'][0, 0].cpu().numpy()
        pred_prob = sample['pred_prob']
        
        gt_x, gt_y = sample['gt_coords']
        pred_x, pred_y = sample['pred_coords']
        distance = sample['distance']
        level_name = sample['level_name']
        
        # Plot image with points
        axes[row, 0].imshow(img[:, :, 0], cmap='gray')
        axes[row, 0].scatter(gt_x, gt_y, c='r', marker='x', s=100, label='Ground Truth')
        axes[row, 0].scatter(pred_x, pred_y, c='b', marker='o', s=100, label='Prediction')
        axes[row, 0].set_title(f"Level {level_name} - Distance: {distance:.2f} px")
        axes[row, 0].legend()
        
        # Plot target heatmap
        axes[row, 1].imshow(target_heatmap, cmap='hot')
        axes[row, 1].set_title("Ground Truth Heatmap")
        
        # Plot predicted heatmap
        axes[row, 2].imshow(pred_prob, cmap='hot')
        axes[row, 2].set_title("Predicted Heatmap")
        
        row += 1
    
    # Add overall title
    plt.suptitle(f"Localization Predictions - Epoch {epoch+1}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['training']['output_dir'], f"canal_localizer_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("train_localizer", os.path.join(output_dir, "train.log"))
    
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
    
    # Create data module - Specifically for localization
    logger.info("Creating localization data module...")
    data_module = CanalLocalizationDataModule(
        data_dir=config['data']['data_dir'],
        canal_slices_file=processed_data['canal_slices'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        target_size=tuple(config['data']['target_size']),
        split_ratio=config['data']['split_ratio'],
        seed=config['data']['seed'],
        only_positive_samples=True
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
    
    # Create model - Specifically for localization
    logger.info("Creating canal localization model...")
    model = SpinalCanalLocalizationModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        in_channels=config['model']['in_channels'],
        dropout_rate=config['model']['dropout_rate']
    )
    model = model.to(device)
    
    # Create loss function - Specifically for localization
    logger.info("Creating localization loss...")
    loss_fn = LocalizationLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        pos_weight=config['training']['loss'].get('pos_weight', None)
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
    
    # Save command line arguments
    with open(os.path.join(output_dir, "args.txt"), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Train
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs...")
    
    # Track best model
    best_val_metrics = {
        'loss': float('inf'),
        'avg_distance': float('inf'),
        'median_distance': float('inf')
    }
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
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=data_module.val_dataloader(),
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            logger=logger,
            visualize=True,
            save_dir=vis_dir
        )
        
        # Update TensorBoard
        writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
        writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
        
        writer.add_scalar("Metrics/Train/AvgDistance", train_metrics['avg_distance'], epoch)
        writer.add_scalar("Metrics/Train/MedianDistance", train_metrics['median_distance'], epoch)
        
        writer.add_scalar("Metrics/Val/AvgDistance", val_metrics['avg_distance'], epoch)
        writer.add_scalar("Metrics/Val/MedianDistance", val_metrics['median_distance'], epoch)
        
        # Add level-specific metrics
        for level_idx in range(5):
            level_name = model.get_level_name(level_idx)
            
            # Training distances
            if level_idx in train_metrics['level_distances']:
                writer.add_scalar(
                    f"Metrics/Train/Level_{level_name}/AvgDistance", 
                    train_metrics['level_distances'][level_idx], 
                    epoch
                )
                writer.add_scalar(
                    f"Metrics/Train/Level_{level_name}/MedianDistance", 
                    train_metrics['level_median_distances'][level_idx], 
                    epoch
                )
            
            # Validation distances
            if level_idx in val_metrics['level_distances']:
                writer.add_scalar(
                    f"Metrics/Val/Level_{level_name}/AvgDistance", 
                    val_metrics['level_distances'][level_idx], 
                    epoch
                )
                writer.add_scalar(
                    f"Metrics/Val/Level_{level_name}/MedianDistance", 
                    val_metrics['level_median_distances'][level_idx], 
                    epoch
                )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['avg_distance'])
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # Check if this is the best model
        is_best = False
        
        if val_metrics['avg_distance'] < best_val_metrics['avg_distance']:
            best_val_metrics['avg_distance'] = val_metrics['avg_distance']
            is_best = True
        
        if val_metrics['median_distance'] < best_val_metrics['median_distance']:
            best_val_metrics['median_distance'] = val_metrics['median_distance']
            is_best = True
        
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics['loss'] = val_metrics['loss']
            is_best = True
        
        if is_best:
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                checkpoint,
                os.path.join(output_dir, "best_localizer.pth")
            )
            logger.info(f"New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(
        checkpoint,
        os.path.join(output_dir, "final_localizer.pth")
    )
    
    # Log best results
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_val_metrics.items()])
    logger.info(f"Best model at epoch {best_epoch+1} with {metrics_str}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
