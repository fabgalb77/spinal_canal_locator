#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script for slice classification model.
This script focuses on evaluating and visualizing the results of the slice classification model
without the canal localization model.
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
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from data.preprocessing import setup_logger
from models.SliceClassificationModel import SliceClassificationModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug the slice classification model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--classifier_checkpoint", 
        type=str, 
        default='best_slice_classifier.pth',
        help="Path to slice classifier model checkpoint"
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
        default="./debug_results",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--gpu", 
        type=int, 
        default=0,
        help="GPU ID to use (-1 for CPU)"
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
        default=10,
        help="Maximum number of series to process"
    )
    
    parser.add_argument(
        "--visualize_all_slices",
        action="store_true",
        help="Create visualizations for all slices in a series (Warning: may generate many images)"
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


def load_and_preprocess_dicom(dicom_path, target_size=(128, 128), include_adjacent=True):
    """
    Load and preprocess a DICOM image for inference.
    
    Args:
        dicom_path: Path to DICOM file
        target_size: Target image size
        include_adjacent: Whether to include adjacent slices for context
        
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
        
        # Create single-channel image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        # Normalize for model
        image = (image - 0.5) / 0.5
        
        # Convert to tensor [C, H, W]
        if include_adjacent:
            # For now, just duplicate the current slice to simulate 3-channel input
            # In a real implementation, you would load adjacent slices here
            image_tensor = torch.tensor(np.concatenate([image] * 3, axis=2).transpose(2, 0, 1), 
                                     dtype=torch.float32).unsqueeze(0)
        else:
            # Single channel expanded to batch dimension
            image_tensor = torch.tensor(image.transpose(2, 0, 1), 
                                     dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, image, (orig_height, orig_width)
        
    except Exception as e:
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None, None, None


def process_series_classification(
    series_path: Path,
    classifier_model: SliceClassificationModel,
    device: torch.device,
    target_size: Tuple[int, int],
    top_n_slices: int,
    output_dir: str,
    logger: logging.Logger,
    study_id: Optional[str] = None,
    series_id: Optional[str] = None,
    optimal_thresholds: Optional[Dict[int, float]] = None,
    visualize_all_slices: bool = False
) -> pd.DataFrame:
    """
    Process a series of DICOM images with the slice classification model.
    
    Args:
        series_path: Path to series directory
        classifier_model: Slice classification model
        device: Device to use
        target_size: Target image size
        top_n_slices: Number of top slices to consider per level
        output_dir: Output directory
        logger: Logger
        study_id: Study ID (optional)
        series_id: Series ID (optional)
        optimal_thresholds: Dictionary mapping level index to optimal threshold (optional)
        visualize_all_slices: Whether to create visualizations for all slices
        
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
    
    # Run classification model on all slices
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
                        'orig_dims': orig_dims,
                        'file_path': str(dicom_file)
                    })
        
        except Exception as e:
            logger.error(f"Error classifying file {dicom_file}: {e}")
            continue
    
    # Identify midsagittal (central) slice for reference
    midsagittal_slice = None
    if all_dicom_files:
        # Sort by instance number to ensure order
        sorted_files = sorted(all_dicom_files, key=lambda x: x['instance_number'])
        # Select the middle slice as midsagittal
        middle_index = len(sorted_files) // 2
        midsagittal_slice = sorted_files[middle_index]
        logger.info(f"Identified midsagittal slice: Instance {midsagittal_slice['instance_number']}")
    
    # Analyze scores for each level
    results = []
    
    # Create debug visualization directory
    visualizations_dir = os.path.join(series_output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Visualize distribution of scores for each level
    for level_idx in range(len(level_names)):
        level_name = level_names[level_idx]
        
        # Extract scores for this level
        scores = [slice_data['score'] for slice_data in all_slice_scores[level_idx]]
        instance_numbers = [slice_data['instance_number'] for slice_data in all_slice_scores[level_idx]]
        
        # Create DataFrame for easier analysis
        level_df = pd.DataFrame({
            'instance_number': instance_numbers,
            'score': scores
        })
        
        # Sort by instance number (slice position)
        level_df = level_df.sort_values('instance_number')
        
        # Plot score distribution
        plt.figure(figsize=(12, 6))
        
        # Plot scores vs slice position
        plt.subplot(1, 2, 1)
        plt.plot(level_df['instance_number'], level_df['score'], marker='o', linestyle='-')
        plt.xlabel('Instance Number (Slice Position)')
        plt.ylabel('Classification Score')
        plt.title(f'Level {level_name} - Scores by Slice Position')
        plt.grid(True)
        
        # Add horizontal line at optimal threshold
        if optimal_thresholds and level_idx in optimal_thresholds:
            plt.axhline(y=optimal_thresholds[level_idx], color='r', linestyle='--', 
                       label=f'Threshold = {optimal_thresholds[level_idx]:.3f}')
            plt.legend()
        
        # Plot histogram of scores
        plt.subplot(1, 2, 2)
        plt.hist(level_df['score'], bins=20, alpha=0.7)
        plt.xlabel('Classification Score')
        plt.ylabel('Frequency')
        plt.title(f'Level {level_name} - Score Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, f"level_{level_name.replace('/', '_')}_score_distribution.png"))
        plt.close()
        
        # Sort slices by classification score (descending)
        sorted_slices = sorted(
            all_slice_scores[level_idx], 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Select top N slices
        top_slices = sorted_slices[:top_n_slices]
        
        # Get level-specific threshold
        level_threshold = 0.5  # Default threshold
        if optimal_thresholds is not None and level_idx in optimal_thresholds:
            level_threshold = optimal_thresholds[level_idx]
        
        # Visualize all slices with their scores if requested
        if visualize_all_slices:
            # Create a mosaic of all slices
            sorted_by_position = sorted(all_slice_scores[level_idx], key=lambda x: x['instance_number'])
            
            # Determine grid size based on number of slices
            num_slices = len(sorted_by_position)
            grid_size = int(np.ceil(np.sqrt(num_slices)))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, slice_data in enumerate(sorted_by_position):
                if i < len(axes):
                    instance_number = slice_data['instance_number']
                    score = slice_data['score']
                    image_np = slice_data['image_np']
                    
                    # Normalize image for display
                    display_img = (image_np[:, :, 0] + 0.5)  # Reverse normalization
                    
                    # Plot image
                    axes[i].imshow(display_img, cmap='gray')
                    axes[i].set_title(f"Inst: {instance_number}\nScore: {score:.3f}")
                    axes[i].axis('off')
                    
                    # Highlight top slices
                    is_top = any(s['instance_number'] == instance_number for s in top_slices)
                    if is_top:
                        axes[i].spines['bottom'].set_color('red')
                        axes[i].spines['top'].set_color('red') 
                        axes[i].spines['right'].set_color('red')
                        axes[i].spines['left'].set_color('red')
                        axes[i].spines['bottom'].set_linewidth(3)
                        axes[i].spines['top'].set_linewidth(3) 
                        axes[i].spines['right'].set_linewidth(3)
                        axes[i].spines['left'].set_linewidth(3)
            
            # Hide empty axes
            for i in range(len(sorted_by_position), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, f"level_{level_name.replace('/', '_')}_all_slices.png"))
            plt.close(fig)
        
        # Log top slices
        for i, slice_data in enumerate(top_slices):
            logger.info(
                f"Level {level_name} - Top {i+1}: Instance {slice_data['instance_number']} "
                f"with score {slice_data['score']:.4f}"
            )
            
            # Add to results
            results.append({
                'study_id': study_id,
                'series_id': series_id,
                'instance_number': slice_data['instance_number'],
                'level': level_name,
                'level_idx': level_idx,
                'rank': i + 1,  # 1-based rank
                'classification_score': slice_data['score'],
                'threshold': level_threshold,
                'passes_threshold': slice_data['score'] >= level_threshold,
                'slice_position': sorted_by_position.index(slice_data) / len(sorted_by_position),  # Relative position
                'file_path': slice_data['file_path']
            })
        
        # Create visualization of top slices
        fig, axes = plt.subplots(1, top_n_slices, figsize=(5 * top_n_slices, 5))
        
        # Handle single slice case
        if top_n_slices == 1:
            axes = [axes]
        
        for i, slice_data in enumerate(top_slices):
            # Get image data
            instance_number = slice_data['instance_number']
            score = slice_data['score']
            image_np = slice_data['image_np']
            
            # Normalize image for display
            display_img = (image_np[:, :, 0] + 0.5)  # Reverse normalization
            
            # Plot image
            axes[i].imshow(display_img, cmap='gray')
            
            # Add score info
            title = f"Instance: {instance_number}\nScore: {score:.4f}"
            if score >= level_threshold:
                title += "\n(Above Threshold)"
                axes[i].set_title(title, color='green')
            else:
                title += "\n(Below Threshold)"
                axes[i].set_title(title, color='red')
            
            # Turn off axis labels
            axes[i].axis('off')
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, f"level_{level_name.replace('/', '_')}_top_{top_n_slices}.png"))
        plt.close(fig)
    
    # Create summary visualization with all levels
    if results:
        # Find best slice for each level
        best_slices = {}
        for level_idx in range(len(level_names)):
            level_name = level_names[level_idx]
            # Filter results for this level and get top ranked
            level_results = [r for r in results if r['level_idx'] == level_idx and r['rank'] == 1]
            if level_results:
                best_slices[level_idx] = level_results[0]
        
        # Create a comparison of best slices across all levels
        fig, axes = plt.subplots(1, len(best_slices), figsize=(5 * len(best_slices), 5))
        
        # Handle single level case
        if len(best_slices) == 1:
            axes = [axes]
        
        # Sort levels by index
        for i, level_idx in enumerate(sorted(best_slices.keys())):
            result = best_slices[level_idx]
            level_name = result['level']
            instance_number = result['instance_number']
            score = result['classification_score']
            
            # Load image
            image_tensor, image_np, _ = load_and_preprocess_dicom(result['file_path'], target_size=target_size)
            
            if image_np is not None:
                # Normalize image for display
                display_img = (image_np[:, :, 0] + 0.5)  # Reverse normalization
                
                # Plot image
                axes[i].imshow(display_img, cmap='gray')
                
                # Add score info
                title = f"Level: {level_name}\nInstance: {instance_number}\nScore: {score:.4f}"
                if result['passes_threshold']:
                    axes[i].set_title(title, color='green')
                else:
                    axes[i].set_title(title, color='red')
                
                # Turn off axis labels
                axes[i].axis('off')
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(os.path.join(series_output_dir, "all_levels_best_slices.png"))
        plt.close(fig)
    
    # Create side view visualization to show where selected slices are in the volume
    if results and len(all_dicom_files) > 0:
        # Sort all files by instance number
        sorted_files = sorted(all_dicom_files, key=lambda x: x['instance_number'])
        
        # Get instance numbers of all slices
        all_instances = [f['instance_number'] for f in sorted_files]
        
        # Find position of top slice for each level
        level_positions = {}
        for level_idx in range(len(level_names)):
            level_name = level_names[level_idx]
            level_results = [r for r in results if r['level_idx'] == level_idx and r['rank'] == 1]
            if level_results:
                instance = level_results[0]['instance_number']
                if instance in all_instances:
                    position = all_instances.index(instance)
                    level_positions[level_name] = position
        
        # Create side view visualization
        plt.figure(figsize=(12, 6))
        
        # Plot positions
        plt.plot(range(len(all_instances)), [0] * len(all_instances), 'k-', linewidth=2)
        
        # Add markers for selected slices
        colors = plt.cm.tab10(np.linspace(0, 1, len(level_names)))
        for i, (level_name, position) in enumerate(level_positions.items()):
            plt.plot(position, 0, 'o', markersize=10, color=colors[i], label=level_name)
        
        plt.yticks([])
        plt.xlabel('Slice Position')
        plt.title('Selected Slices in Volume')
        plt.legend()
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(series_output_dir, "slice_positions.png"))
        plt.close()
    
    # Create DataFrame from results
    if not results:
        logger.warning(f"No results for series {study_id}/{series_id}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Write results to CSV
    results_csv = os.path.join(series_output_dir, "classification_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    logger.info(f"Saved results to {results_csv}")
    
    return results_df


def main():
    """Main function for slice classification model debugging."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"classifier_debug_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger("classifier_debug", os.path.join(output_dir, "debug.log"))
    
    # Set device
    device = torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.info("Using CPU")
    
    # Create classifier model
    logger.info("Creating slice classification model...")
    classifier_model = SliceClassificationModel(
        backbone=config['model']['backbone'],
        pretrained=False,  # Don't need pretrained weights for inference
        in_channels=config['model']['in_channels'],
        dropout_rate=0.0  # No dropout for inference
    )
    
    # Load checkpoint
    logger.info(f"Loading classifier checkpoint from {args.classifier_checkpoint}")
    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in classifier_checkpoint:
        classifier_model.load_state_dict(classifier_checkpoint['model_state_dict'])
    else:
        classifier_model.load_state_dict(classifier_checkpoint)
    
    # Load or define optimal thresholds per level
    if 'optimal_thresholds' in classifier_checkpoint:
        logger.info("Using optimal thresholds from checkpoint")
        optimal_thresholds = classifier_checkpoint['optimal_thresholds']
    else:
        # Default optimal thresholds based on typical validation results
        logger.info("Using default optimal thresholds")
        optimal_thresholds = {
            0: 0.25,  # L1/L2
            1: 0.20,  # L2/L3
            2: 0.30,  # L3/L4
            3: 0.25,  # L4/L5
            4: 0.25,  # L5/S1
        }
    
    logger.info(f"Optimal thresholds per level: {optimal_thresholds}")
    
    # Move model to device
    classifier_model = classifier_model.to(device)
    
    # Set model to evaluation mode
    classifier_model.eval()
    
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
            results = process_series_classification(
                series_path=series_dir,
                classifier_model=classifier_model,
                device=device,
                target_size=target_size,
                top_n_slices=args.top_n_slices,
                output_dir=output_dir,
                logger=logger,
                study_id=study_id,
                series_id=series_dir.name,
                optimal_thresholds=optimal_thresholds,
                visualize_all_slices=args.visualize_all_slices
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
        results_path = os.path.join(output_dir, "combined_classification_results.csv")
        combined_results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Log summary statistics
        logger.info(f"Processed {processed_series_count} series")
        logger.info(f"Classified {len(combined_results)} slice selections")
        
        # Create a summary table by level
        level_stats = combined_results.groupby('level').agg({
            'study_id': 'nunique',
            'series_id': 'nunique',
            'classification_score': ['mean', 'min', 'max', 'count'],
            'passes_threshold': ['sum', 'mean']
        })
        
        logger.info("\nLevel-specific statistics:")
        logger.info(level_stats)
        
        # Analyze score distributions
        plt.figure(figsize=(12, 8))
        
        # Plot score distributions by level
        for i, level_name in enumerate(classifier_model.level_names):
            level_data = combined_results[combined_results['level'] == level_name]
            if not level_data.empty:
                plt.subplot(len(classifier_model.level_names), 1, i+1)
                sns.histplot(level_data['classification_score'], kde=True, bins=30)
                plt.title(f"Level {level_name} Score Distribution")
                plt.xlabel("Classification Score")
                plt.ylabel("Count")
                
                # Add vertical line at threshold
                if i in optimal_thresholds:
                    plt.axvline(x=optimal_thresholds[i], color='r', linestyle='--',
                              label=f'Threshold: {optimal_thresholds[i]:.3f}')
                    plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_levels_score_distributions.png"))
        plt.close()
        
        # Plot relative slice positions
        plt.figure(figsize=(12, 8))
        
        # Plot slice position distributions by level
        for i, level_name in enumerate(classifier_model.level_names):
            level_data = combined_results[(combined_results['level'] == level_name) & 
                                      (combined_results['rank'] == 1)]  # Only top ranked slices
            if not level_data.empty:
                plt.subplot(len(classifier_model.level_names), 1, i+1)
                sns.histplot(level_data['slice_position'], kde=True, bins=20)
                plt.title(f"Level {level_name} Slice Position Distribution")
                plt.xlabel("Relative Position in Volume (0=Start, 1=End)")
                plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_levels_position_distributions.png"))
        plt.close()
    else:
        logger.warning("No results found")
    
    logger.info("Debugging complete")


if __name__ == "__main__":
    main()