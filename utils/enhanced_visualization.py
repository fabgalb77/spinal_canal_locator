"""
Enhanced visualization functions for multi-headed spinal canal model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any


def visualize_train_samples(
    model, 
    train_dataloader, 
    device, 
    output_dir,
    epoch,
    num_samples=3
):
    """
    Visualize predictions on training samples.
    
    Args:
        model: Model to use for predictions
        train_dataloader: Training dataloader
        device: Device to use
        output_dir: Directory to save visualizations
        epoch: Current epoch
        num_samples: Number of samples to visualize
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    batch_iter = iter(train_dataloader)
    batch = next(batch_iter)
    
    # Create a figure with one row for each sample, three columns (image, target, prediction)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 5 * num_samples))
    
    # Process each sample
    for i in range(min(num_samples, len(batch['image']))):
        # Get data for this sample
        image = batch['image'][i:i+1].to(device)
        heatmap = batch['heatmap'][i:i+1].to(device)
        level_idx = batch['level_idx'][i].item()
        level_name = batch['level'][i]
        
        # Get scaled coordinates
        scaled_x, scaled_y = batch['scaled_coordinates'][0][i].item(), batch['scaled_coordinates'][1][i].item()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(image, level_idx=level_idx)
            
            # Get predictions for all levels too
            all_outputs = model.forward_all_levels(image)
        
        # Denormalize image
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5) + 0.5  # Denormalize
        
        # Get target heatmap
        target_heatmap = heatmap[0].cpu().numpy()
        
        # Get predicted heatmap
        pred_heatmap = outputs[0, 0].cpu().numpy()
        pred_prob = 1 / (1 + np.exp(-pred_heatmap))  # Apply sigmoid
        
        # Find predicted peak coordinates
        pred_y, pred_x = np.unravel_index(pred_prob.argmax(), pred_prob.shape)
        
        # Calculate distance
        distance = np.sqrt((pred_x - scaled_x) ** 2 + (pred_y - scaled_y) ** 2)
        
        # Plot image with points
        axes[i, 0].imshow(img_np[:, :, 0], cmap='gray')
        axes[i, 0].scatter(scaled_x, scaled_y, c='r', marker='x', s=100, label='Ground Truth')
        axes[i, 0].scatter(pred_x, pred_y, c='b', marker='o', s=100, label='Prediction')
        axes[i, 0].set_title(f"Train Sample {i+1} - Level {level_name}")
        axes[i, 0].legend()
        
        # Plot target heatmap
        axes[i, 1].imshow(target_heatmap, cmap='hot')
        axes[i, 1].set_title("Ground Truth Heatmap")
        
        # Plot predicted heatmap
        axes[i, 2].imshow(pred_prob, cmap='hot')
        axes[i, 2].set_title(f"Predicted Heatmap - Distance: {distance:.2f} px")
        
        # Plot predictions from other level heads
        for j in range(2):
            # Get prediction from a different level head
            other_level_idx = (level_idx + j + 1) % 5
            other_level_name = model.get_level_name(other_level_idx)
            other_pred = all_outputs[0, other_level_idx].cpu().numpy()
            other_prob = 1 / (1 + np.exp(-other_pred))
            
            # Plot
            axes[i, j+3].imshow(other_prob, cmap='hot')
            axes[i, j+3].set_title(f"Level {other_level_name} Head Output")
    
    # Add overall title
    plt.suptitle(f"Training Samples - Epoch {epoch+1}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f"train_samples_epoch_{epoch+1}.png"))
    plt.close(fig)


def visualize_val_samples(
    model, 
    val_dataloader, 
    device, 
    output_dir,
    epoch,
    num_samples=3
):
    """
    Visualize predictions on validation samples.
    
    Args:
        model: Model to use for predictions
        val_dataloader: Validation dataloader
        device: Device to use
        output_dir: Directory to save visualizations
        epoch: Current epoch
        num_samples: Number of samples to visualize
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    batch_iter = iter(val_dataloader)
    batch = next(batch_iter)
    
    # Create a figure with one row for each sample, three columns (image, target, prediction)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 5 * num_samples))
    
    # Process each sample
    for i in range(min(num_samples, len(batch['image']))):
        # Get data for this sample
        image = batch['image'][i:i+1].to(device)
        heatmap = batch['heatmap'][i:i+1].to(device)
        level_idx = batch['level_idx'][i].item()
        level_name = batch['level'][i]
        
        # Get scaled coordinates
        scaled_x, scaled_y = batch['scaled_coordinates'][0][i].item(), batch['scaled_coordinates'][1][i].item()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(image, level_idx=level_idx)
            
            # Get predictions for all levels too
            all_outputs = model.forward_all_levels(image)
        
        # Denormalize image
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 0.5) + 0.5  # Denormalize
        
        # Get target heatmap
        target_heatmap = heatmap[0].cpu().numpy()
        
        # Get predicted heatmap
        pred_heatmap = outputs[0, 0].cpu().numpy()
        pred_prob = 1 / (1 + np.exp(-pred_heatmap))  # Apply sigmoid
        
        # Find predicted peak coordinates
        pred_y, pred_x = np.unravel_index(pred_prob.argmax(), pred_prob.shape)
        
        # Calculate distance
        distance = np.sqrt((pred_x - scaled_x) ** 2 + (pred_y - scaled_y) ** 2)
        
        # Plot image with points
        axes[i, 0].imshow(img_np[:, :, 0], cmap='gray')
        axes[i, 0].scatter(scaled_x, scaled_y, c='r', marker='x', s=100, label='Ground Truth')
        axes[i, 0].scatter(pred_x, pred_y, c='b', marker='o', s=100, label='Prediction')
        axes[i, 0].set_title(f"Val Sample {i+1} - Level {level_name}")
        axes[i, 0].legend()
        
        # Plot target heatmap
        axes[i, 1].imshow(target_heatmap, cmap='hot')
        axes[i, 1].set_title("Ground Truth Heatmap")
        
        # Plot predicted heatmap
        axes[i, 2].imshow(pred_prob, cmap='hot')
        axes[i, 2].set_title(f"Predicted Heatmap - Distance: {distance:.2f} px")
        
        # Plot predictions from other level heads
        for j in range(2):
            # Get prediction from a different level head
            other_level_idx = (level_idx + j + 1) % 5
            other_level_name = model.get_level_name(other_level_idx)
            other_pred = all_outputs[0, other_level_idx].cpu().numpy()
            other_prob = 1 / (1 + np.exp(-other_pred))
            
            # Plot
            axes[i, j+3].imshow(other_prob, cmap='hot')
            axes[i, j+3].set_title(f"Level {other_level_name} Head Output")
    
    # Add overall title
    plt.suptitle(f"Validation Samples - Epoch {epoch+1}", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f"val_samples_epoch_{epoch+1}.png"))
    plt.close(fig)


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
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize training samples
    visualize_train_samples(
        model=model,
        train_dataloader=train_dataloader,
        device=device,
        output_dir=vis_dir,
        epoch=epoch,
        num_samples=num_samples
    )
    
    # Visualize validation samples
    visualize_val_samples(
        model=model,
        val_dataloader=val_dataloader,
        device=device,
        output_dir=vis_dir,
        epoch=epoch,
        num_samples=num_samples
    )
