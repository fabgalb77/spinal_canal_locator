#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for dual-headed spinal canal localization.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import cv2
import pydicom
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from data.preprocessing import setup_logger
from models.DualHeadCanalModel import DualHeadCanalModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with dual-headed canal model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Path to input DICOM directory"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./inference_results",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0,
        help="GPU ID to use (-1 for CPU)"
    )
    
    parser.add_argument(
        "--classification_threshold", 
        type=float, 
        default=0.8,
        help="Confidence threshold for classification"
    )
    
    parser.add_argument(
        "--localization_threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for localization heatmap"
    )
    
    parser.add_argument(
        "--combined_threshold", 
        type=float, 
        default=0.7,
        help="Threshold for combined confidence score (geometric mean)"
    )
    
    parser.add_argument(
        "--max_series", 
        type=int, 
        default=None,
        help="Maximum number of series to process"
    )
    
    return parser.parse_args()


def is_sagittal_series(dicom_dataset):
    """Determine if a DICOM series is sagittal based on metadata."""
    try:
        # Check series description
        series_description = getattr(dicom_dataset, 'SeriesDescription', '').lower()
        description_is_sagittal = 'sagittal' in series_description
        
        # Check image orientation
        if hasattr(dicom_dataset, 'ImageOrientationPatient'):
            orientation = dicom_dataset.ImageOrientationPatient
            if len(orientation) >= 6:
                # Simplified check for sagittal orientation
                is_sagittal_by_orientation = (
                    abs(orientation[0]) < 0.5 and 
                    abs(abs(orientation[1]) - 1) < 0.5 and 
                    abs(orientation[2]) < 0.5
                )
                return is_sagittal_by_orientation
        
        return description_is_sagittal
        
    except Exception as e:
        print(f"Error determining if series is sagittal: {e}")
        return False


def load_and_preprocess_dicom(dicom_path, target_size=(128, 128)):
    """Load and preprocess a DICOM image for inference."""
    try:
        # Read DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        image = dicom.pixel_array.astype(np.float32)
        
        # Store original dimensions
        orig_height, orig_width = image.shape
        
        # Normalize to [0, 1]
        if image.max() > 0:
            image = image / image.max()
        
        # Resize
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Convert to RGB (3 channels)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Normalize for model
        image = (image - 0.5) / 0.5
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, image, (orig_height, orig_width)
        
    except Exception as e:
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None, None, None


def process_series(
    series_path,
    model,
    device,
    target_size,
    cls_threshold,
    loc_threshold,
    combined_threshold,
    output_dir,
    logger,
    study_id=None,
    series_id=None
):
    """
    Process a series of DICOM images with the dual-headed model.
    
    Args:
        series_path: Path to series directory
        model: Model for inference
        device: Device to use
        target_size: Target image size
        cls_threshold: Confidence threshold for classification
        loc_threshold: Confidence threshold for localization
        combined_threshold: Threshold for combined confidence
        output_dir: Output directory
        logger: Logger
        study_id: Study ID (optional)
        series_id: Series ID (optional)
        
    Returns:
        DataFrame with results
    """
    # Extract study/series IDs from path if not provided
    if study_id is None:
        study_id = series_path.parent.name
    if series_id is None:
        series_id = series_path.name
    
    logger.info(f"Processing series {study_id}/{series_id}")
    
    # Get DICOM files
    dicom_files = sorted(list(series_path.glob("*.dcm")))
    
    if not dicom_files:
        logger.warning(f"No DICOM files found in {series_path}")
        return pd.DataFrame()
    
    # Check if the first image is sagittal
    try:
        first_dicom = pydicom.dcmread(str(dicom_files[0]))
        if not is_sagittal_series(first_dicom):
            logger.info(f"Skipping non-sagittal series: {study_id}/{series_id}")
            return pd.DataFrame()
        logger.info(f"Confirmed sagittal series: {study_id}/{series_id}")
    except Exception as e:
        logger.error(f"Error reading first DICOM file: {e}")
        return pd.DataFrame()
    
    # Create output directory for this series
    series_output_dir = os.path.join(output_dir, study_id, series_id)
    os.makedirs(series_output_dir, exist_ok=True)
    
    # Initialize results
    results = []
    
    # Level names from model
    level_names = model.level_names
    
    # Track best detection per level
    best_detections = {level_name: {'confidence': 0, 'data': None} for level_name in level_names}
    
    # Process each DICOM file
    for dicom_file in tqdm(dicom_files, desc=f"Processing {study_id}/{series_id}"):
        try:
            # Get instance number
            instance_number = dicom_file.stem
            
            # Load and preprocess DICOM
            image_tensor, image_np, orig_dims = load_and_preprocess_dicom(
                dicom_file, target_size=target_size
            )
            
            if image_tensor is None:
                continue
            
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Run inference for all levels
            with torch.no_grad():
                # Process all levels at once
                heatmaps, classifications = model.forward_all_levels(image_tensor, apply_sigmoid=True)
                
                # Create a figure for visualization showing all levels
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                # Show the original image in the first subplot
                axes[0].imshow(image_np[:, :, 0], cmap='gray')
                axes[0].set_title(f"Original - Instance {instance_number}")
                
                # Initialize detected points
                detected_points = []
                
                # Process each level
                for level_idx, level_name in enumerate(level_names):
                    # Get outputs for this level
                    heatmap = heatmaps[0, level_idx].cpu().numpy()
                    cls_score = classifications[0, level_idx].item()
                    
                    # Find peak and its value in the heatmap
                    y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                    loc_score = heatmap.max()
                    
                    # Calculate combined confidence score (geometric mean)
                    combined_confidence = (cls_score * loc_score) ** 0.5
                    
                    # Only process if all thresholds are met
                    if (cls_score > cls_threshold and 
                        loc_score > loc_threshold and 
                        combined_confidence > combined_threshold):
                        
                        # Add to results
                        result = {
                            'study_id': study_id,
                            'series_id': series_id,
                            'instance_number': instance_number,
                            'level': level_name,
                            'level_idx': level_idx,
                            'x': int(x),
                            'y': int(y),
                            'classification_confidence': cls_score,
                            'localization_confidence': loc_score,
                            'combined_confidence': combined_confidence
                        }
                        results.append(result)
                        
                        # Add detection for visualization
                        detected_points.append((x, y, level_name, combined_confidence))
                        
                        # Check if this is the best detection for this level
                        if combined_confidence > best_detections[level_name]['confidence']:
                            best_detections[level_name]['confidence'] = combined_confidence
                            best_detections[level_name]['data'] = {
                                'image': image_np.copy(),
                                'heatmap': heatmap.copy(),
                                'instance_number': instance_number,
                                'x': x,
                                'y': y,
                                'cls_score': cls_score,
                                'loc_score': loc_score,
                                'combined_confidence': combined_confidence
                            }
                        
                        # Log finding
                        logger.info(
                            f"Found canal at level {level_name} in instance {instance_number} "
                            f"at coordinates ({x}, {y}) with confidence {combined_confidence:.3f} "
                            f"(Cls: {cls_score:.3f}, Loc: {loc_score:.3f})"
                        )
                    
                    # Always add to visualization (with alpha based on confidence)
                    if level_idx + 1 < len(axes):
                        # Show heatmap overlay with alpha based on classification confidence
                        alpha = cls_score * 0.9  # Scale alpha by classification confidence
                        
                        # Show the original image
                        axes[level_idx + 1].imshow(image_np[:, :, 0], cmap='gray')
                        
                        # Overlay heatmap
                        im = axes[level_idx + 1].imshow(
                            heatmap, 
                            cmap='hot', 
                            alpha=alpha
                        )
                        
                        # Add colorbar
                        plt.colorbar(im, ax=axes[level_idx + 1])
                        
                        # Add title with confidence scores
                        axes[level_idx + 1].set_title(
                            f"Level {level_name}\n"
                            f"Cls: {cls_score:.3f}, Loc: {loc_score:.3f}, "
                            f"Combined: {combined_confidence:.3f}"
                        )
                        
                        # Add marker if detection is valid
                        if combined_confidence > combined_threshold:
                            axes[level_idx + 1].scatter(x, y, c='b', marker='x', s=100)
                
                # Show detected points on the original image
                for x, y, level_name, conf in detected_points:
                    axes[0].scatter(x, y, c='r', marker='x', s=100)
                    axes[0].text(
                        x + 5, y + 5, 
                        f"{level_name}: {conf:.2f}", 
                        color='yellow', 
                        bbox=dict(facecolor='black', alpha=0.5)
                    )
                
                # Hide any unused subplots
                for i in range(len(level_names) + 1, len(axes)):
                    axes[i].axis('off')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(series_output_dir, f"{instance_number}_all_levels.png"))
                plt.close(fig)
                
        except Exception as e:
            logger.error(f"Error processing file {dicom_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Create a summary visualization of best detections for each level
    if any(detection['data'] is not None for detection in best_detections.values()):
        fig, axes = plt.subplots(len(level_names), 3, figsize=(15, 5 * len(level_names)))
        
        # Handle case with only one level detected
        if len(level_names) == 1:
            axes = axes.reshape(1, 3)
        
        # Process each level
        for i, level_name in enumerate(level_names):
            detection = best_detections[level_name]
            
            if detection['data'] is not None:
                data = detection['data']
                
                # Plot original image with detection point
                axes[i, 0].imshow(data['image'][:, :, 0], cmap='gray')
                axes[i, 0].scatter(data['x'], data['y'], c='r', marker='x', s=100)
                axes[i, 0].set_title(
                    f"Level {level_name} - Instance {data['instance_number']}\n"
                    f"Combined Confidence: {data['combined_confidence']:.3f}"
                )
                
                # Plot heatmap
                im = axes[i, 1].imshow(data['heatmap'], cmap='hot')
                plt.colorbar(im, ax=axes[i, 1])
                axes[i, 1].set_title(f"Heatmap (Max: {data['loc_score']:.3f})")
                
                # Plot overlay
                axes[i, 2].imshow(data['image'][:, :, 0], cmap='gray')
                im = axes[i, 2].imshow(data['heatmap'], cmap='hot', alpha=0.7)
                axes[i, 2].scatter(data['x'], data['y'], c='b', marker='x', s=100)
                axes[i, 2].set_title(f"Overlay - Cls Score: {data['cls_score']:.3f}")
            else:
                # No detection for this level
                for j in range(3):
                    axes[i, j].text(
                        0.5, 0.5, 
                        f"No detection for {level_name}", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[i, j].transAxes
                    )
                    axes[i, j].axis('off')
        
        # Save the summary figure
        plt.tight_layout()
        plt.savefig(os.path.join(series_output_dir, "best_detections_summary.png"))
        plt.close(fig)
    
    return pd.DataFrame(results)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dual_head_inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("inference_dual_head", os.path.join(output_dir, "inference.log"))
    
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
        pretrained=False,  # Don't need pretrained weights for inference
        in_channels=config['model']['in_channels'],
        dropout_rate=0.0  # No dropout for inference
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get target size
    target_size = tuple(config['data']['target_size'])
    
    # Log threshold settings
    logger.info(f"Classification threshold: {args.classification_threshold}")
    logger.info(f"Localization threshold: {args.localization_threshold}")
    logger.info(f"Combined threshold: {args.combined_threshold}")
    
    # Process input directory
    input_path = Path(args.input_dir)
    
    # List all patient directories
    study_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(study_dirs)} studies")
    
    # Limit number of series if specified
    processed_series_count = 0
    
    # Initialize results
    all_results = []
    
    # Process each study
    for study_dir in study_dirs:
        study_id = study_dir.name
        logger.info(f"Processing study: {study_id}")
        
        # List all series for this study
        series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(series_dirs)} series for study {study_id}")
        
        # Process each series
        for series_dir in series_dirs:
            # Check if we've processed enough series
            if args.max_series is not None and processed_series_count >= args.max_series:
                logger.info(f"Reached maximum number of series ({args.max_series})")
                break
            
            # Process this series
            results = process_series(
                series_path=series_dir,
                model=model,
                device=device,
                target_size=target_size,
                cls_threshold=args.classification_threshold,
                loc_threshold=args.localization_threshold,
                combined_threshold=args.combined_threshold,
                output_dir=output_dir,
                logger=logger,
                study_id=study_id,
                series_id=series_dir.name
            )
            
            # Add to overall results
            if not results.empty:
                all_results.append(results)
            
            # Increment counter
            processed_series_count += 1
        
        # Check if we've processed enough series
        if args.max_series is not None and processed_series_count >= args.max_series:
            break
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save results to CSV
        results_path = os.path.join(output_dir, "dual_head_inference_results.csv")
        combined_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Log summary statistics
        logger.info(f"Processed {processed_series_count} series")
        logger.info(f"Found {len(combined_results)} canal locations")
        
        # Log level-specific statistics
        level_counts = combined_results.groupby('level').size()
        logger.info("Level-specific detection counts:")
        for level, count in level_counts.items():
            logger.info(f"  {level}: {count}")
            
        # Log confidence statistics
        logger.info(f"Mean combined confidence: {combined_results['combined_confidence'].mean():.4f}")
        logger.info(f"Median combined confidence: {combined_results['combined_confidence'].median():.4f}")
        logger.info(f"Mean classification confidence: {combined_results['classification_confidence'].mean():.4f}")
        logger.info(f"Mean localization confidence: {combined_results['localization_confidence'].mean():.4f}")
    else:
        logger.warning("No results found")
    
    logger.info("Inference complete")


if __name__ == "__main__":
    main()
