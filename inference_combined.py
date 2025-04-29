#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for combined slice classification and canal localization.
This script integrates both models into a single pipeline:
1. Slice classification model selects the optimal slice for each spinal level
2. Canal localization model then finds the precise canal location within that slice
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from models.SliceClassificationModel import SliceClassificationModel
from models.SpinalCanalLocalizationModel import SpinalCanalLocalizationModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with combined spinal canal models")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--classifier_checkpoint", 
        type=str, 
        default='final_slice_classifier.pth',
        help="Path to slice classifier model checkpoint"
    )
    
    parser.add_argument(
        "--localizer_checkpoint", 
        type=str, 
        default='final_localizer.pth',
        help="Path to canal localizer model checkpoint"
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default='../rsna-2024-lumbar-spine-degenerative-classification/test_images',
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
        default=0.1,
        help="Confidence threshold for classification"
    )
    
    parser.add_argument(
        "--top_n_slices", 
        type=int, 
        default=3,
        help="Number of top slices to consider per level"
    )
    
    parser.add_argument(
        "--max_series", 
        type=int, 
        default=None,
        help="Maximum number of series to process"
    )
    
    return parser.parse_args()


def is_sagittal_series(dicom_dataset):
    """
    Determine if a DICOM series is sagittal based on metadata.
    
    Args:
        dicom_dataset: DICOM dataset
        
    Returns:
        True if series is likely sagittal, False otherwise
    """
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
    """
    Load and preprocess a DICOM image for inference.
    
    Args:
        dicom_path: Path to DICOM file
        target_size: Target image size
        
    Returns:
        Tuple of (image_tensor, image_np, original_dimensions)
    """
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
    series_path: Path,
    classifier_model: SliceClassificationModel,
    localizer_model: SpinalCanalLocalizationModel,
    device: torch.device,
    target_size: Tuple[int, int],
    classification_threshold: float,
    top_n_slices: int,
    output_dir: str,
    logger: logging.Logger,
    study_id: Optional[str] = None,
    series_id: Optional[str] = None,
    optimal_thresholds: Optional[Dict[int, float]] = None
) -> pd.DataFrame:
    """
    Process a series of DICOM images with the combined slice classification and canal localization models.
    
    Args:
        series_path: Path to series directory
        classifier_model: Slice classification model
        localizer_model: Canal localization model
        device: Device to use
        target_size: Target image size
        classification_threshold: Default confidence threshold for classification
        top_n_slices: Number of top slices to consider per level
        output_dir: Output directory
        logger: Logger
        study_id: Study ID (optional)
        series_id: Series ID (optional)
        optimal_thresholds: Dictionary mapping level index to optimal threshold (optional)
        
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
    
    # Level names from model
    level_names = classifier_model.level_names
    
    # Initialize storage for slice classification results
    all_slices_classification = {}  # Dict to store classification scores for all slices and levels
    
    # Step 1: Run classification model on all slices to find optimal slice for each level
    logger.info(f"Classifying slices for series {study_id}/{series_id}...")
    
    # Initialize results
    all_slice_scores = {level_idx: [] for level_idx in range(len(level_names))}
    all_dicom_files = []  # Track all valid DICOM files
    
    # Process each DICOM file for classification
    for dicom_file in tqdm(dicom_files, desc=f"Classifying slices in {study_id}/{series_id}"):
        try:
            # Get instance number
            instance_number = int(dicom_file.stem)
            
            # Load and preprocess DICOM
            image_tensor, image_np, orig_dims = load_and_preprocess_dicom(
                dicom_file, target_size=target_size
            )
            
            if image_tensor is None:
                continue
            
            # Store valid DICOM file
            all_dicom_files.append({
                'file': dicom_file, 
                'instance_number': instance_number,
                'image_tensor': image_tensor,
                'image_np': image_np,
                'orig_dims': orig_dims
            })
            
            # Move to device
            image_tensor = image_tensor.to(device)
            
            # Run classification for all levels
            with torch.no_grad():
                # Get classification scores for all levels
                classification_scores = classifier_model.forward_all_levels(image_tensor, apply_sigmoid=True)
                
                # Store scores for each level
                for level_idx in range(len(level_names)):
                    score = classification_scores[0, level_idx].item()
                    all_slice_scores[level_idx].append({
                        'instance_number': instance_number,
                        'score': score,
                        'image_tensor': image_tensor,
                        'image_np': image_np,
                        'orig_dims': orig_dims
                    })
        
        except Exception as e:
            logger.error(f"Error classifying file {dicom_file}: {e}")
            continue
    
    # Identify midsagittal (central) slice for fallback
    midsagittal_slice = None
    if all_dicom_files:
        # Sort by instance number to ensure order
        sorted_files = sorted(all_dicom_files, key=lambda x: x['instance_number'])
        # Select the middle slice as midsagittal
        middle_index = len(sorted_files) // 2
        midsagittal_slice = sorted_files[middle_index]
        logger.info(f"Identified midsagittal slice: Instance {midsagittal_slice['instance_number']}")
    
    
    # Step 2: For each level, select top N slices with highest classification scores
    logger.info(f"Selecting top {top_n_slices} slices for each level...")
    
    top_slices = {}  # Dict to store top slices for each level
    
    for level_idx in range(len(level_names)):
        level_name = level_names[level_idx]
        
        # Sort slices by classification score (descending)
        sorted_slices = sorted(
            all_slice_scores[level_idx], 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Select top N slices
        top_slices[level_idx] = sorted_slices[:top_n_slices]
        
        # Log top slices
        for i, slice_data in enumerate(top_slices[level_idx]):
            logger.info(
                f"Level {level_name} - Top {i+1}: Instance {slice_data['instance_number']} "
                f"with score {slice_data['score']:.4f}"
            )
    
    # Step 3: Run localization model on top slices to find canal locations
    logger.info(f"Localizing canals in top slices...")
    
    # Initialize results
    results = []
    
    # Process each level
    for level_idx in range(len(level_names)):
        level_name = level_names[level_idx]
        
        # Get level-specific threshold
        level_threshold = classification_threshold
        if optimal_thresholds is not None and level_idx in optimal_thresholds:
            level_threshold = optimal_thresholds[level_idx]

        # Check if we have any good slices for this level
        if not top_slices[level_idx] or top_slices[level_idx][0]['score'] < level_threshold:
            logger.warning(f"No good slices found for level {level_name} (threshold: {level_threshold})")
            
            # Use midsagittal slice as fallback if available
            if midsagittal_slice is not None:
                logger.info(f"Using midsagittal slice (instance {midsagittal_slice['instance_number']}) as fallback for level {level_name}")
                
                # Create a fallback slice entry with the midsagittal slice
                fallback_slice = {
                    'instance_number': midsagittal_slice['instance_number'],
                    'score': 0.0,  # Mark with score 0 to indicate fallback
                    'image_tensor': midsagittal_slice['image_tensor'],
                    'image_np': midsagittal_slice['image_np'],
                    'orig_dims': midsagittal_slice['orig_dims']
                }
                
                # Process the fallback slice
                instance_number = fallback_slice['instance_number']
                image_tensor = fallback_slice['image_tensor'].to(device)
                image_np = fallback_slice['image_np']
                orig_height, orig_width = fallback_slice['orig_dims']
                
                # Run localization model
                with torch.no_grad():
                    # Get heatmap for this level
                    heatmap = localizer_model(image_tensor, level_idx=level_idx)
                    
                    # Convert heatmap to probability map
                    heatmap_np = heatmap[0, 0].cpu().numpy()
                    prob_map = 1 / (1 + np.exp(-heatmap_np))  # sigmoid
                    
                    # Find peak in heatmap
                    y, x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                    
                    # Get localization confidence (peak value)
                    loc_score = prob_map.max()
                    
                    # Rescale coordinates to original image dimensions
                    orig_x = int(x * (orig_width / target_size[1]))
                    orig_y = int(y * (orig_height / target_size[0]))
                
                # Create result entry
                result = {
                    'study_id': study_id,
                    'series_id': series_id,
                    'instance_number': instance_number,
                    'level': level_name,
                    'level_idx': level_idx,
                    'rank': 1,  # Mark as top rank
                    'x': x,
                    'y': y,
                    'orig_x': orig_x,
                    'orig_y': orig_y,
                    'classification_score': fallback_slice['score'],
                    'classification_threshold': level_threshold,
                    'localization_score': loc_score,
                    'is_fallback': True  # Flag to indicate this is a fallback slice
                }
                
                results.append(result)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Show original image
                axes[0].imshow(image_np[:, :, 0], cmap='gray')
                axes[0].scatter(x, y, c='r', marker='x', s=100)
                axes[0].set_title(f"Level {level_name} - Instance {instance_number}\nFALLBACK MIDSAGITTAL SLICE")
                
                # Show heatmap
                axes[1].imshow(prob_map, cmap='hot')
                axes[1].set_title(f"Localization Heatmap\nPeak Value: {loc_score:.4f}")
                
                # Show overlay
                axes[2].imshow(image_np[:, :, 0], cmap='gray')
                axes[2].imshow(prob_map, cmap='hot', alpha=0.5)
                axes[2].scatter(x, y, c='b', marker='x', s=100)
                axes[2].set_title(f"Overlay\nCoordinates: ({x}, {y})")
                
                # Save visualization
                plt.tight_layout()
                # Replace slashes in level name to avoid directory confusion
                safe_level_name = level_name.replace('/', '_')
                plt.savefig(os.path.join(series_output_dir, f"level_{safe_level_name}_instance_{instance_number}_FALLBACK.png"))
                plt.close(fig)
                
                logger.info(
                    f"Using fallback midsagittal slice for level {level_name} in instance {instance_number} "
                    f"at coordinates ({x}, {y}) with localization score {loc_score:.4f}"
                )
            continue
        
        # Process top slices for this level
        for rank, slice_data in enumerate(top_slices[level_idx]):
            instance_number = slice_data['instance_number']
            cls_score = slice_data['score']
            
            # Skip slices below threshold
            if cls_score < level_threshold:
                continue
            
            # Get image data
            image_tensor = slice_data['image_tensor']
            image_np = slice_data['image_np']
            orig_height, orig_width = slice_data['orig_dims']
            
            # Run localization model
            with torch.no_grad():
                # Get heatmap for this level
                heatmap = localizer_model(image_tensor, level_idx=level_idx)
                
                # Convert heatmap to probability map
                heatmap_np = heatmap[0, 0].cpu().numpy()
                prob_map = 1 / (1 + np.exp(-heatmap_np))  # sigmoid
                
                # Find peak in heatmap
                y, x = np.unravel_index(prob_map.argmax(), prob_map.shape)
                
                # Get localization confidence (peak value)
                loc_score = prob_map.max()
                
                # Rescale coordinates to original image dimensions
                orig_x = int(x * (orig_width / target_size[1]))
                orig_y = int(y * (orig_height / target_size[0]))
            
            # Create result entry
            result = {
                'study_id': study_id,
                'series_id': series_id,
                'instance_number': instance_number,
                'level': level_name,
                'level_idx': level_idx,
                'rank': rank + 1,  # 1-based rank
                'x': x,
                'y': y,
                'orig_x': orig_x,
                'orig_y': orig_y,
                'classification_score': cls_score,
                'classification_threshold': level_threshold,
                'localization_score': loc_score
            }
            
            results.append(result)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Show original image
            axes[0].imshow(image_np[:, :, 0], cmap='gray')
            axes[0].scatter(x, y, c='r', marker='x', s=100)
            axes[0].set_title(f"Level {level_name} - Instance {instance_number}\nClassification Score: {cls_score:.4f}")
            
            # Show heatmap
            axes[1].imshow(prob_map, cmap='hot')
            axes[1].set_title(f"Localization Heatmap\nPeak Value: {loc_score:.4f}")
            
            # Show overlay
            axes[2].imshow(image_np[:, :, 0], cmap='gray')
            axes[2].imshow(prob_map, cmap='hot', alpha=0.5)
            axes[2].scatter(x, y, c='b', marker='x', s=100)
            axes[2].set_title(f"Overlay\nCoordinates: ({x}, {y})")
            
            # Save visualization
            plt.tight_layout()
            # Replace slashes in level name to avoid directory confusion
            safe_level_name = level_name.replace('/', '_')
            plt.savefig(os.path.join(series_output_dir, f"level_{safe_level_name}_instance_{instance_number}.png"))
            plt.close(fig)
            
            logger.info(
                f"Found canal for level {level_name} in instance {instance_number} "
                f"at coordinates ({x}, {y}) with classification score {cls_score:.4f} "
                f"and localization score {loc_score:.4f}"
            )

    # Create DataFrame from results
    if not results:
        logger.warning(f"No results for series {study_id}/{series_id}")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Add a column to indicate fallback status if not present
    if 'is_fallback' not in results_df.columns:
        results_df['is_fallback'] = False
    
    # Write results to CSV
    results_csv = os.path.join(series_output_dir, "results.csv")
    results_df.to_csv(results_csv, index=False)
    
    logger.info(f"Saved results to {results_csv}")

    # Create summary visualization with all levels
    if results:
        # Group results by level
        level_results = {}
        for result in results:
            level_idx = result['level_idx']
            if level_idx not in level_results or result['rank'] == 1:  # Keep only top-ranked slice
                level_results[level_idx] = result
        
        # Create figure with all levels
        fig, axes = plt.subplots(len(level_results), 2, figsize=(12, 5 * len(level_results)))
        
        # Handle case with only one level
        if len(level_results) == 1:
            axes = axes.reshape(1, 2)
        
        # Sort levels by index
        sorted_level_indices = sorted(level_results.keys())
        
        for i, level_idx in enumerate(sorted_level_indices):
            result = level_results[level_idx]
            level_name = result['level']
            instance_number = result['instance_number']
            
            # Load the DICOM image for visualization
            dicom_path = os.path.join(series_path, f"{instance_number}.dcm")
            image_tensor, image_np, _ = load_and_preprocess_dicom(dicom_path, target_size=target_size)
            
            if image_np is None:
                continue
            
            # Get coordinates
            x, y = result['x'], result['y']
            
            # Show original image
            axes[i, 0].imshow(image_np[:, :, 0], cmap='gray')
            axes[i, 0].scatter(x, y, c='r', marker='x', s=100)
            axes[i, 0].set_title(
                f"Level {level_name} - Instance {instance_number}\n"
                f"Classification Score: {result['classification_score']:.4f}"
            )
            
            # Run localization model to get heatmap
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                heatmap = localizer_model(image_tensor, level_idx=level_idx)
                heatmap_np = heatmap[0, 0].cpu().numpy()
                prob_map = 1 / (1 + np.exp(-heatmap_np))
            
            # Show overlay
            axes[i, 1].imshow(image_np[:, :, 0], cmap='gray')
            axes[i, 1].imshow(prob_map, cmap='hot', alpha=0.5)
            axes[i, 1].scatter(x, y, c='b', marker='x', s=100)
            axes[i, 1].set_title(
                f"Canal Localization\n"
                f"Coordinates: ({x}, {y})"
            )
        
        # Save summary visualization
        plt.tight_layout()
        plt.savefig(os.path.join(series_output_dir, "all_levels_summary.png"))
        plt.close(fig)
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def main():
    """Main function for combined slice classification and canal localization inference."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"combined_inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("inference_combined", os.path.join(output_dir, "inference.log"))
    
    # Set device
    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("Using CPU")
    
    # Create models
    logger.info("Creating slice classification model...")
    classifier_model = SliceClassificationModel(
        backbone=config['model']['backbone'],
        pretrained=False,  # Don't need pretrained weights for inference
        in_channels=config['model']['in_channels'],
        dropout_rate=0.0  # No dropout for inference
    )
    
    logger.info("Creating canal localization model...")
    localizer_model = SpinalCanalLocalizationModel(
        backbone=config['model']['backbone'],
        pretrained=False,  # Don't need pretrained weights for inference
        in_channels=config['model']['in_channels'],
        dropout_rate=0.0  # No dropout for inference
    )
    
    # Load checkpoints
    logger.info(f"Loading classifier checkpoint from {args.classifier_checkpoint}")
    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location=device, weights_only=False)
    
    logger.info(f"Loading localizer checkpoint from {args.localizer_checkpoint}")
    localizer_checkpoint = torch.load(args.localizer_checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in classifier_checkpoint:
        classifier_model.load_state_dict(classifier_checkpoint['model_state_dict'])
    else:
        classifier_model.load_state_dict(classifier_checkpoint)
    
    if 'model_state_dict' in localizer_checkpoint:
        localizer_model.load_state_dict(localizer_checkpoint['model_state_dict'])
    else:
        localizer_model.load_state_dict(localizer_checkpoint)
    
    # Load or define optimal thresholds per level
    # These should ideally come from validation during training
    if 'optimal_thresholds' in classifier_checkpoint:
        logger.info("Using optimal thresholds from checkpoint")
        optimal_thresholds = classifier_checkpoint['optimal_thresholds']
    else:
        # Default optimal thresholds based on typical validation results
        # These are better than a single fixed threshold
        logger.info("Using default optimal thresholds")
        optimal_thresholds = {
            0: 0.25,  # L1/L2
            1: 0.20,  # L2/L3
            2: 0.30,  # L3/L4
            3: 0.25,  # L4/L5
            4: 0.25,  # L5/S1
        }
    
    logger.info(f"Optimal thresholds per level: {optimal_thresholds}")
    
    # Move models to device
    classifier_model = classifier_model.to(device)
    localizer_model = localizer_model.to(device)
    
    # Set models to evaluation mode
    classifier_model.eval()
    localizer_model.eval()
    
    # Get target size
    target_size = tuple(config['data']['target_size'])
    
    # Process input directory
    input_path = Path(args.input_dir)
    
    # List all patient directories
    study_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(study_dirs)} studies")
    
    # Initialize results
    all_results = []
    
    # Limit number of series if specified
    processed_series_count = 0
    
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
                classifier_model=classifier_model,
                localizer_model=localizer_model,
                device=device,
                target_size=target_size,
                classification_threshold=args.classification_threshold,
                top_n_slices=args.top_n_slices,
                output_dir=output_dir,
                logger=logger,
                study_id=study_id,
                series_id=series_dir.name,
                optimal_thresholds=optimal_thresholds  # Pass optimal thresholds
            )
            
            # Add to overall results
            if not results.empty:
                all_results.append(results)
            
            # Increment counter
            processed_series_count += 1
        
        # Check if we've processed enough series
        if args.max_series is not None and processed_series_count >= args.max_series:
            break
            
            # Process this series
            results = process_series(
                series_path=series_dir,
                classifier_model=classifier_model,
                localizer_model=localizer_model,
                device=device,
                target_size=target_size,
                classification_threshold=args.classification_threshold,
                top_n_slices=args.top_n_slices,
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
        results_path = os.path.join(output_dir, "combined_inference_results.csv")
        combined_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Log summary statistics
        logger.info(f"Processed {processed_series_count} series")
        logger.info(f"Found {len(combined_results)} canal locations")
        
        # Create a summary table by level
        level_stats = combined_results.groupby('level').agg({
            'study_id': 'nunique',
            'series_id': 'nunique',
            'instance_number': 'count',
            'classification_score': ['mean', 'min', 'max'],
            'localization_score': ['mean', 'min', 'max']
        })
        
        logger.info("\nLevel-specific statistics:")
        logger.info(level_stats)
    else:
        logger.warning("No results found")
    
    logger.info("Inference complete")


if __name__ == "__main__":
    main()
