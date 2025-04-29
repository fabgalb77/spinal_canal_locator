#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the slice classification model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve

from data.preprocessing import create_processed_dataset, setup_logger
from data.SliceClassificationDataset import SliceClassificationDataset, SliceClassificationDataModule
from models.SliceClassificationModel import SliceClassificationModel


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
    parser = argparse.ArgumentParser(description="Train slice classification model")
    
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
        "--negative_ratio",
        type=float,
        default=3.0,
        help="Ratio of negative examples to include"
    )
    
    parser.add_argument(
        "--hard_negative_mining",
        action="store_true",
        help="Use hard negative mining"
    )
    
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        help="Use focal loss instead of BCE"
    )
    
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=3.0,
        help="Positive class weight for BCE loss"
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
        reduction: str = "mean",
        pos_weight: float = 2.0,
        label_smoothing: float = 0.05  # Added label smoothing
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
            pos_weight: Weight for positive examples
            label_smoothing: Amount of label smoothing to apply
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.eps = 1e-6
    
    def forward(self, inputs, targets):
        """Forward pass with label smoothing."""
        # Apply label smoothing
        smoothed_targets = targets.clone()
        smoothed_targets[targets == 1] = 1.0 - self.label_smoothing
        smoothed_targets[targets == 0] = self.label_smoothing
        
        # Apply sigmoid to get probabilities
        inputs_prob = torch.sigmoid(inputs)
        inputs_prob = torch.clamp(inputs_prob, self.eps, 1.0 - self.eps)
        
        # Calculate BCE with pos_weight
        weight = self.pos_weight * smoothed_targets + (1 - smoothed_targets)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction="none", 
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
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
        Average training loss
    """
    model.train()
    running_loss = 0.0
    
    # Track predictions and targets for metrics
    all_preds = []
    all_targets = []
    
    # Track level-specific losses
    level_losses = {}
    level_counts = {}
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # Get data
        images = batch["image"].to(device)
        is_optimal = batch["is_optimal"].to(device)
        level_indices = batch["level_idx"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Initialize batch loss
        batch_loss = 0.0
        
        # Group samples by level
        unique_levels = torch.unique(level_indices).cpu().numpy()
        
        # Process each level group
        for level_idx in unique_levels:
            # Get samples for this level
            level_mask = level_indices == level_idx
            level_images = images[level_mask]
            level_targets = is_optimal[level_mask]
            
            # Skip if no samples for this level (shouldn't happen)
            if level_images.size(0) == 0:
                continue
                
            # Forward pass for this level
            predictions = model(level_images, level_idx=int(level_idx))
            
            # Calculate loss
            loss = loss_fn(predictions, level_targets)
            
            # Accumulate loss (weighted by number of samples)
            batch_loss += loss * level_mask.sum().item()
            
            # Track level-specific loss
            if level_idx not in level_losses:
                level_losses[level_idx] = 0.0
                level_counts[level_idx] = 0
            
            level_losses[level_idx] += loss.item() * level_mask.sum().item()
            level_counts[level_idx] += level_mask.sum().item()
            
            # Track predictions and targets for metrics
            pred_probs = torch.sigmoid(predictions).cpu().detach().numpy().flatten()
            target_vals = level_targets.cpu().detach().numpy().flatten()
            
            all_preds.extend(pred_probs.tolist())
            all_targets.extend(target_vals.tolist())
            
            # Backward pass for this level group
            loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Normalize batch loss by total number of samples
        batch_loss = batch_loss / images.size(0)
        
        # Update running loss
        running_loss += batch_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": batch_loss.item()})
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    try:
        auc = roc_auc_score(all_targets, all_preds)
        ap = average_precision_score(all_targets, all_preds)
        
        # Calculate binary predictions for confusion matrix
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f}, "
            f"AUC: {auc:.4f}, AP: {ap:.4f}, "
            f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}"
        )
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.6f}")
    
    # Log level-specific losses
    for level_idx, loss_sum in level_losses.items():
        count = level_counts[level_idx]
        level_name = model.get_level_name(level_idx)
        avg_level_loss = loss_sum / count if count > 0 else 0
        logger.info(f"  Level {level_name}: Loss: {avg_level_loss:.6f} ({count} samples)")
    
    return avg_loss


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
        Average validation loss and metrics
    """
    model.eval()
    running_loss = 0.0
    
    # Track predictions and targets for metrics
    all_preds = []
    all_targets = []
    level_preds_dict = {i: [] for i in range(5)}
    level_targets_dict = {i: [] for i in range(5)}
    
    # Track samples for visualization
    vis_samples = {i: [] for i in range(5)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]"):
            # Get data
            images = batch["image"].to(device)
            is_optimal = batch["is_optimal"].to(device)
            level_indices = batch["level_idx"].to(device)
            
            # Group samples by level
            unique_levels = torch.unique(level_indices).cpu().numpy()
            
            # Process each level group
            for level_idx in unique_levels:
                # Get samples for this level
                level_mask = level_indices == level_idx
                level_images = images[level_mask]
                level_targets = is_optimal[level_mask]
                level_idx = int(level_idx)
                
                # Skip if no samples for this level (shouldn't happen)
                if level_images.size(0) == 0:
                    continue
                    
                # Forward pass for this level
                predictions = model(level_images, level_idx=level_idx)
                
                # Calculate loss
                loss = loss_fn(predictions, level_targets)
                
                # Update running loss
                running_loss += loss.item() * level_mask.sum().item()
                
                # Track predictions and targets for metrics
                pred_probs = torch.sigmoid(predictions).cpu().detach().numpy().flatten()
                target_vals = level_targets.cpu().detach().numpy().flatten()
                
                all_preds.extend(pred_probs.tolist())
                all_targets.extend(target_vals.tolist())
                
                level_preds_dict[level_idx].extend(pred_probs.tolist())
                level_targets_dict[level_idx].extend(target_vals.tolist())
                
                # Store samples for visualization
                if visualize:
                    for i in range(min(level_images.size(0), 2)):
                        # Only store up to 2 samples per level
                        if len(vis_samples[level_idx]) >= 2:
                            break
                            
                        # Get sample data
                        img = level_images[i:i+1]
                        target_val = level_targets[i].item()
                        pred_prob = pred_probs[i]
                        
                        # Store one positive and one negative example for each level
                        is_pos = target_val > 0.5
                        
                        # Check if we already have a sample of this type
                        have_pos = any(v['target'] > 0.5 for v in vis_samples[level_idx])
                        have_neg = any(v['target'] <= 0.5 for v in vis_samples[level_idx])
                        
                        if (is_pos and not have_pos) or (not is_pos and not have_neg):
                            # Get the original index in the batch
                            orig_idx = torch.nonzero(level_mask)[i].item()
                            
                            vis_samples[level_idx].append({
                                'image': img.cpu(),
                                'target': target_val,
                                'pred': pred_prob,
                                'level_idx': level_idx,
                                'level_name': batch["level"][orig_idx]
                            })
        
        # Calculate average loss
        avg_loss = running_loss / len(dataloader.dataset)

        # Calculate overall metrics
        try:
            auc = roc_auc_score(all_targets, all_preds)
            ap = average_precision_score(all_targets, all_preds)
            
            # Find optimal threshold using validation data
            # This is a critical fix to prevent the "all positive/negative" issue
            fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Use optimal threshold instead of fixed 0.5
            binary_preds = [1 if p > optimal_threshold else 0 for p in all_preds]
            tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = sensitivity
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Log metrics with threshold information
            logger.info(
                f"Epoch {epoch+1} - Val Loss: {avg_loss:.6f}, AUC: {auc:.4f}, AP: {ap:.4f}, "
                f"Optimal Threshold: {optimal_threshold:.4f}, "
                f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
                f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}"
            )

            # Calculate level-specific metrics
            level_metrics = {}
            
            for level_idx in range(5):
                if level_preds_dict[level_idx] and level_targets_dict[level_idx]:
                    try:
                        level_auc = roc_auc_score(level_targets_dict[level_idx], level_preds_dict[level_idx])
                        
                        # Calculate level-specific optimal threshold
                        fpr, tpr, level_thresholds = roc_curve(level_targets_dict[level_idx], level_preds_dict[level_idx])
                        level_optimal_idx = np.argmax(tpr - fpr)
                        level_optimal_threshold = level_thresholds[level_optimal_idx]
                        
                        # Calculate accuracy with optimal threshold
                        level_acc = sum(1 for p, t in zip(level_preds_dict[level_idx], level_targets_dict[level_idx]) 
                                    if (p > level_optimal_threshold) == (t > 0.5)) / len(level_preds_dict[level_idx])
                        
                        level_metrics[level_idx] = {
                            'auc': level_auc,
                            'accuracy': level_acc,
                            'threshold': level_optimal_threshold
                        }
                        
                        level_name = model.get_level_name(level_idx)
                        logger.info(
                            f"  Level {level_name}: AUC: {level_auc:.4f}, Accuracy: {level_acc:.4f} "
                            f"(Threshold: {level_optimal_threshold:.4f}, {len(level_preds_dict[level_idx])} samples)"
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating metrics for level {level_idx}: {e}")
            
            # Visualize predictions
            if visualize and save_dir:
                visualize_predictions(
                    vis_samples=vis_samples,
                    save_path=os.path.join(save_dir, f"val_viz_epoch_{epoch+1}.png"),
                    epoch=epoch
                )
            
            return {
                'loss': avg_loss,
                'auc': auc,
                'ap': ap,
                'accuracy': accuracy,
                'f1': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'level_metrics': level_metrics
            }
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.6f}")
            return {'loss': avg_loss}


def visualize_predictions(vis_samples, save_path, epoch):
    """
    Visualize model predictions.
    
    Args:
        vis_samples: Dictionary of visualization samples
        save_path: Path to save visualization
        epoch: Current epoch
    """
    # Count total number of samples
    total_samples = sum(len(samples) for samples in vis_samples.values())
    
    if total_samples == 0:
        return
    
    # Create figure with one row per sample
    fig, axes = plt.subplots(total_samples, 1, figsize=(10, 5 * total_samples))
    
    # Handle case with just one sample
    if total_samples == 1:
        axes = [axes]
    
    # Plot samples
    sample_idx = 0
    
    for level_idx in range(5):
        for sample in vis_samples[level_idx]:
            # Get data
            img = sample['image'][0].cpu().numpy().transpose(1, 2, 0)
            img = (img * 0.5) + 0.5  # Denormalize
            target = sample['target']
            pred = sample['pred']
            level_name = sample['level_name']
            
            # Plot image
            axes[sample_idx].imshow(img[:, :, 0], cmap='gray')
            
            # Add title with prediction info
            title = f"Level {level_name} - "
            title += f"True: {'Optimal' if target > 0.5 else 'Not Optimal'}, "
            title += f"Pred: {'Optimal' if pred > 0.5 else 'Not Optimal'} ({pred:.4f})"
            
            axes[sample_idx].set_title(title)
            
            # Increment counter
            sample_idx += 1
    
    # Add overall title
    plt.suptitle(f"Validation Predictions - Epoch {epoch+1}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)  # Make sure to close the figure


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
    output_dir = os.path.join(config['training']['output_dir'], f"slice_classifier_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("train_slice_classifier", os.path.join(output_dir, "train.log"))
    
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
    
    # Create data module - Specifically for classification
    logger.info("Creating classification data module...")
    data_module = SliceClassificationDataModule(
        data_dir=config['data']['data_dir'],
        canal_slices_file=processed_data['canal_slices'],
        batch_size=config['data']['batch_size'] * 2,  # Larger batch size for classification
        num_workers=1,  # Reduce workers to avoid multiprocessing issues with matplotlib
        target_size=tuple(config['data']['target_size']),
        split_ratio=config['data']['split_ratio'],
        seed=config['data']['seed'],
        include_negatives=True,
        negative_ratio=args.negative_ratio,
        hard_negative_mining=args.hard_negative_mining,
        series_sampling=True
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
    

    # Create model - Specifically for classification
    logger.info("Creating slice classification model...")
    model = SliceClassificationModel(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        in_channels=config['model']['in_channels'],
        dropout_rate=config['model']['slice_classifier']['dropout_rate']
    )
    model = model.to(device)

    # Create loss function
    if args.focal_loss:
        logger.info("Using Focal Loss with label smoothing...")
        loss_fn = FocalLoss(pos_weight=args.pos_weight, label_smoothing=0.05)
    else:
        logger.info(f"Using BCE Loss with pos_weight={args.pos_weight} and label smoothing...")
        
        # Create a custom BCE loss with label smoothing
        class BCEWithLabelSmoothing(nn.Module):
            def __init__(self, pos_weight, smoothing=0.05):
                super().__init__()
                self.pos_weight = pos_weight
                self.smoothing = smoothing
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
                
            def forward(self, pred, target):
                # Apply label smoothing
                smoothed_target = target.clone()
                smoothed_target[target == 1] = 1.0 - self.smoothing
                smoothed_target[target == 0] = self.smoothing
                
                # Calculate loss with pos_weight
                weight = torch.ones_like(smoothed_target)
                weight[smoothed_target > 0.5] = self.pos_weight
                
                loss = self.criterion(pred, smoothed_target)
                loss = (loss * weight).mean()
                return loss
                
        loss_fn = BCEWithLabelSmoothing(pos_weight=args.pos_weight)
    
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
        'auc': 0.0,
        'accuracy': 0.0,
        'f1': 0.0
    }
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train for one epoch
        train_loss = train_one_epoch(
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
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
        
        if 'auc' in val_metrics:
            writer.add_scalar("Metrics/Val/AUC", val_metrics['auc'], epoch)
        if 'ap' in val_metrics:
            writer.add_scalar("Metrics/Val/AP", val_metrics['ap'], epoch)
        if 'accuracy' in val_metrics:
            writer.add_scalar("Metrics/Val/Accuracy", val_metrics['accuracy'], epoch)
        if 'f1' in val_metrics:
            writer.add_scalar("Metrics/Val/F1", val_metrics['f1'], epoch)
        if 'sensitivity' in val_metrics:
            writer.add_scalar("Metrics/Val/Sensitivity", val_metrics['sensitivity'], epoch)
        if 'specificity' in val_metrics:
            writer.add_scalar("Metrics/Val/Specificity", val_metrics['specificity'], epoch)
        
        # Add level-specific metrics to TensorBoard
        if 'level_metrics' in val_metrics:
            for level_idx, metrics in val_metrics['level_metrics'].items():
                level_name = model.get_level_name(level_idx)
                
                if 'auc' in metrics:
                    writer.add_scalar(f"Metrics/Val/Level_{level_name}/AUC", metrics['auc'], epoch)
                if 'accuracy' in metrics:
                    writer.add_scalar(f"Metrics/Val/Level_{level_name}/Accuracy", metrics['accuracy'], epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        
        # Extract optimal thresholds from validation metrics
        level_optimal_thresholds = {}
        if 'level_metrics' in val_metrics:
            for level_idx, metrics in val_metrics['level_metrics'].items():
                if 'threshold' in metrics:
                    level_optimal_thresholds[level_idx] = metrics['threshold']

        # Create checkpoint with optimal thresholds
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "optimal_thresholds": level_optimal_thresholds  # Add optimal thresholds
        }
        
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # Check if this is the best model
        is_best = False
        
        if 'auc' in val_metrics and val_metrics['auc'] > best_val_metrics['auc']:
            best_val_metrics['auc'] = val_metrics['auc']
            is_best = True
        
        if 'accuracy' in val_metrics and val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics['accuracy'] = val_metrics['accuracy']
            is_best = True
        
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics['loss'] = val_metrics['loss']
            is_best = True
        
        if 'f1' in val_metrics and val_metrics['f1'] > best_val_metrics['f1']:
            best_val_metrics['f1'] = val_metrics['f1']
            is_best = True
        
        if is_best:
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(
                checkpoint,
                os.path.join(output_dir, "best_slice_classifier.pth")
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
        os.path.join(output_dir, "final_slice_classifier.pth")
    )
    
    # Log best results
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_val_metrics.items()])
    logger.info(f"Best model at epoch {best_epoch+1} with {metrics_str}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
