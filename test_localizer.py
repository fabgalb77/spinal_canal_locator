#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for spinal canal localization model.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Import from existing files
from models.SpinalCanalLocalizationModel import SpinalCanalLocalizationModel
from data.preprocessing import setup_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test spinal canal localization model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--test_dir", 
        type=str, 
        default="/mnt/c/users/fabio/Dropbox/RSNA/rsna-2024-lumbar-spine-degenerative-classification/test_images",
        help="Path to test data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./test_results",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0,
        help="GPU ID to use (-1 for CPU)"
    )
    
    return parser.parse_args()


def is_sagittal_series(dicom_path: str) -> bool:
    """
    Determine if a series is sagittal based on DICOM metadata.
    Uses multiple methods to detect sagittal orientation.
    
    Args:
        dicom_path: Path to a DICOM file in the series
        
    Returns:
        True if the series is sagittal, False otherwise
    """
    try:
        # Load DICOM file
        dcm = pydicom.dcmread(dicom_path)
        
        # Method 1: Check series description (case insensitive)
        if hasattr(dcm, 'SeriesDescription'):
            description = dcm.SeriesDescription.lower()
            if 'sag' in description:
                return True
        
        # Method 2: Check image orientation patient
        if hasattr(dcm, 'ImageOrientationPatient'):
            orientation = dcm.ImageOrientationPatient
            
            # For sagittal images, the X component (left-right axis) of both
            # row and column direction cosines should be near 0
            # Being more lenient with the threshold (0.5 instead of 0.3)
            if abs(orientation[0]) < 0.5 and abs(orientation[3]) < 0.5:
                return True
        
        # Method 3: Check image type for SAG indicator
        if hasattr(dcm, 'ImageType'):
            image_type = ' '.join(dcm.ImageType).lower()
            if 'sag' in image_type:
                return True
        
        # Method 4: Check protocol name if available
        if hasattr(dcm, 'ProtocolName'):
            protocol = dcm.ProtocolName.lower()
            if 'sag' in protocol:
                return True
                
        # Method 5: Check image position patient values
        # Sagittal series often have constant X coordinates across slices
        if hasattr(dcm, 'ImagePositionPatient'):
            # We can't check multiple slices here, but include for future reference
            pass
            
        return False
    
    except Exception as e:
        print(f"Error checking if series is sagittal: {e}")
        return False


def find_midsagittal_slice(series_path: str) -> Optional[str]:
    """
    Find the midsagittal slice in a series.
    
    Args:
        series_path: Path to the directory containing DICOM files
        
    Returns:
        Path to the midsagittal slice DICOM file or None if not found
    """
    try:
        # List all DICOM files in the directory
        dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
        
        if not dicom_files:
            return None
        
        # Sort files by instance number (filename)
        dicom_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Select middle slice
        mid_idx = len(dicom_files) // 2
        mid_slice = dicom_files[mid_idx]
        
        return os.path.join(series_path, mid_slice)
    
    except Exception as e:
        print(f"Error finding midsagittal slice: {e}")
        return None


def load_and_preprocess_image(dicom_path: str, target_size: Tuple[int, int] = (128, 128)):
    """
    Load and preprocess a DICOM image using the same preprocessing as in training.
    
    Args:
        dicom_path: Path to DICOM file
        target_size: Target image size
        
    Returns:
        Preprocessed image tensor and original image
    """
    try:
        # Read DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        image = dicom.pixel_array.astype(np.float32)
        
        # Normalize to [0, 1] properly handling negative values
        if image.max() > image.min():  # Check there's some variation in the image
            image = (image - image.min()) / (image.max() - image.min())
        else:
            # If image is constant (max = min), set to 0.5
            image = np.ones_like(image) * 0.5
        
        # Save original image for visualization
        original_image = image.copy()
        
        # Convert to RGB (3 channels)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Apply transforms - matching training preprocessing
        transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
        transformed = transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_image
    
    except Exception as e:
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None, None


def visualize_all_levels(model, image_tensor, original_image, save_path, device):
    """
    Predict spinal canal locations for all levels and create visualization.
    
    Args:
        model: Trained localization model
        image_tensor: Preprocessed image tensor
        original_image: Original image before transforms
        save_path: Path to save visualization
        device: Device to use for inference
    """
    # Move tensor to device
    image_tensor = image_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model.forward_all_levels(image_tensor, apply_sigmoid=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot each level prediction
    for i, level_name in enumerate(model.level_names):
        # Get heatmap for this level
        heatmap = outputs[0, i].cpu().numpy()
        
        # Find peak location
        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Resize heatmap to match original image size for visualization
        from skimage.transform import resize
        resized_heatmap = resize(
            heatmap, 
            original_image.shape, 
            order=1, 
            preserve_range=True
        )
        
        # Plot original image
        axes[i+1].imshow(original_image, cmap='gray')
        
        # Overlay heatmap
        axes[i+1].imshow(resized_heatmap, cmap='hot', alpha=0.3)
        
        # Calculate peak position in original image space
        orig_height, orig_width = original_image.shape
        orig_peak_x = peak_x * (orig_width / heatmap.shape[1])
        orig_peak_y = peak_y * (orig_height / heatmap.shape[0])
        
        # Mark the peak
        axes[i+1].scatter(orig_peak_x, orig_peak_y, c='r', marker='x', s=100, linewidths=2)
        axes[i+1].set_title(f"Level {level_name}")
        axes[i+1].axis('off')
    
    # Add overall title
    plt.suptitle(f"Spinal Canal Localization", fontsize=16)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("test_localizer", os.path.join(args.output_dir, "test.log"))
    
    # Set device
    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("Using CPU")
    
    # Create model
    model = SpinalCanalLocalizationModel(
        backbone=config['model']['backbone'],
        pretrained=False,  # No need for pretrained weights during inference
        in_channels=config['model']['in_channels'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Load model weights
    model_path = args.model_path or os.path.join(config['training']['output_dir'], "best_localizer.pth")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    
    # Process test directory
    logger.info(f"Processing test data from {args.test_dir}")
    
    # Track statistics
    total_studies = 0
    total_series = 0
    sagittal_series = 0
    processed_series = 0
    
    # Walk through directory structure
    for study_id in tqdm(os.listdir(args.test_dir), desc="Processing studies"):
        study_path = os.path.join(args.test_dir, study_id)
        
        if not os.path.isdir(study_path):
            continue
            
        total_studies += 1
        
        # Create output directory for this study
        study_output_dir = os.path.join(args.output_dir, study_id)
        os.makedirs(study_output_dir, exist_ok=True)
        
        # Process each series in this study
        for series_id in os.listdir(study_path):
            series_path = os.path.join(study_path, series_id)
            
            if not os.path.isdir(series_path):
                continue
                
            total_series += 1
            
            # Try to find a DICOM file to check if it's sagittal
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            
            if not dicom_files:
                logger.warning(f"No DICOM files found in {series_path}")
                continue
                
            # Check first DICOM file to determine if sagittal
            sample_dicom = os.path.join(series_path, dicom_files[0])
            
            if not is_sagittal_series(sample_dicom):
                logger.info(f"Skipping non-sagittal series: {study_id}/{series_id}")
                continue
                
            sagittal_series += 1
            
            # Find midsagittal slice
            midsagittal_path = find_midsagittal_slice(series_path)
            
            if not midsagittal_path:
                logger.warning(f"Could not find midsagittal slice for {study_id}/{series_id}")
                continue
                
            # Load and preprocess image
            image_tensor, original_image = load_and_preprocess_image(
                midsagittal_path, 
                target_size=tuple(config['data']['target_size'])
            )
            
            if image_tensor is None:
                logger.warning(f"Failed to load image: {midsagittal_path}")
                continue
                
            # Generate visualization
            save_path = os.path.join(study_output_dir, f"{series_id}_canal_localization.png")
            visualize_all_levels(model, image_tensor, original_image, save_path, device)
            
            processed_series += 1
            logger.info(f"Processed {study_id}/{series_id} - saved to {save_path}")
    
    # Log summary statistics
    logger.info(f"Testing completed:")
    logger.info(f"  Total studies: {total_studies}")
    logger.info(f"  Total series: {total_series}")
    logger.info(f"  Sagittal series: {sagittal_series}")
    logger.info(f"  Successfully processed: {processed_series}")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()