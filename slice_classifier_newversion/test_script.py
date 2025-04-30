#!/usr/bin/env python3
"""
Enhanced debug script for spine classifier model.
Examines both pre-sigmoid and post-sigmoid values and checks image preprocessing.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from custom modules
from model import SimpleSpineModel, ResNetSpineModel, LEVEL_NAMES
from dataset import TestDataset, is_sagittal_series, create_data_transforms
from image_utils import get_dcm_stats, plot_preprocessing_steps, visualize_results_with_debug

def predict_series_with_debug(model, series_dir, device, output_dir, target_size=(256, 256), batch_size=16):
    """Run predictions on a series and return results with debug info."""
    # Create transforms
    _, test_transform = create_data_transforms(target_size=target_size)
    
    # Get all DICOM files in the series
    all_files = sorted([
        os.path.join(series_dir, f) for f in os.listdir(series_dir) 
        if f.endswith('.dcm')
    ], key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Filter to keep only the central slices
    if len(all_files) > 5:
        middle_idx = len(all_files) // 2
        start_idx = max(0, middle_idx - 2)
        end_idx = min(len(all_files), middle_idx + 3)
        central_files = all_files[start_idx:end_idx]
        print(f"Keeping only {len(central_files)} central slices from {len(all_files)} total slices")
    else:
        # If there are 5 or fewer slices, keep all of them
        central_files = all_files
        print(f"Series has only {len(all_files)} slices, keeping all")
    
    # Create dataset with only central files
    class CentralSlicesDataset(TestDataset):
        def __init__(self, files, target_size=(256, 256), transform=None):
            self.files = files
            self.target_size = target_size
            self.transform = transform
            
            # Create default transform if none provided
            if self.transform is None:
                _, self.transform = create_data_transforms(target_size=target_size)
    
    # Create dataset
    dataset = CentralSlicesDataset(central_files, target_size=target_size, transform=test_transform)
    
    if len(dataset) == 0:
        print(f"No DICOM files found in {series_dir}")
        return None
    
    # Get some DICOM stats for debugging
    first_dicom = dataset.files[0]
    middle_dicom = dataset.files[len(dataset.files)//2]
    
    # Debug preprocessing on first and middle slices
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    study_series = os.path.basename(os.path.dirname(os.path.dirname(series_dir))) + "_" + os.path.basename(series_dir)
    
    first_stats = get_dcm_stats(first_dicom)
    print(f"\nFirst DICOM stats for {study_series}:")
    for k, v in first_stats.items():
        print(f"  {k}: {v}")
    
    # Plot preprocessing steps
    first_debug_path = os.path.join(debug_dir, f"{study_series}_first_preproc.png")
    plot_preprocessing_steps(first_dicom, first_debug_path)
    print(f"Saved preprocessing debug to: {first_debug_path}")
    
    middle_debug_path = os.path.join(debug_dir, f"{study_series}_middle_preproc.png")
    plot_preprocessing_steps(middle_dicom, middle_debug_path)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize results
    results = []
    
    # Run predictions
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            images = batch["image"].to(device)
            instance_numbers = batch["instance_number"]
            dicom_paths = batch["dicom_path"]
            pixel_arrays = batch["pixel_array"]
            
            # Examine image tensor statistics
            img_stats = {
                "min": float(images.min()),
                "max": float(images.max()),
                "mean": float(images.mean()),
                "std": float(images.std())
            }
            print(f"Batch tensor stats: {img_stats}")
            
            # Run predictions for all levels with debug
            all_logits = []
            all_probs = []
            
            for level_idx in range(len(LEVEL_NAMES)):
                logits, probs = model(images, level_idx, return_logits=True)
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
            
            # Print some logit statistics
            for i, level in enumerate(LEVEL_NAMES):
                logit_mean = float(np.mean(all_logits[i]))
                logit_min = float(np.min(all_logits[i]))
                logit_max = float(np.max(all_logits[i]))
                print(f"Level {level} logits: mean={logit_mean:.2f}, min={logit_min:.2f}, max={logit_max:.2f}")
            
            # Store results
            for i in range(len(instance_numbers)):
                if instance_numbers[i] == -1:  # Skip error cases
                    continue
                
                # Store both logits and probabilities for debugging
                level_logits = {
                    LEVEL_NAMES[j]: float(all_logits[j][i].item()) for j in range(len(LEVEL_NAMES))
                }
                
                level_preds = {
                    LEVEL_NAMES[j]: float(all_probs[j][i].item()) for j in range(len(LEVEL_NAMES))
                }
                
                results.append({
                    "instance_number": instance_numbers[i].item(),
                    "dicom_path": dicom_paths[i],
                    "pixel_array": pixel_arrays[i],
                    "logits": level_logits,
                    "predictions": level_preds,
                    "max_pred": max(level_preds.values()),
                    "avg_pred": sum(level_preds.values()) / len(level_preds)
                })
    
    # Sort results by instance number
    results = sorted(results, key=lambda x: x["instance_number"])
    
    # Create a CSV with both logits and probabilities
    if results:
        debug_csv_path = os.path.join(debug_dir, f"{study_series}_debug.csv")
        debug_df = pd.DataFrame([
            {
                "instance_number": r["instance_number"],
                **{f"{level}_logit": r["logits"][level] for level in LEVEL_NAMES},
                **{f"{level}_prob": r["predictions"][level] for level in LEVEL_NAMES}
            }
            for r in results
        ])
        debug_df.to_csv(debug_csv_path, index=False)
        print(f"Saved debug CSV to: {debug_csv_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Debug spine slice classifier on DICOM data')
    parser.add_argument('--test_dir', type=str, default="/mnt/c/users/Fabio Galbusera/Dropbox/RSNA/rsna-2024-lumbar-spine-degenerative-classification/test_images", help='Directory with DICOM files')
    parser.add_argument('--model_path', type=str, default='./output/best_model_simple.pth', help='Path to trained model weights')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'resnet'], help='Model architecture: "simple" or "resnet"')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--image_size', type=int, default=64, help='Image size for model input')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--force_all', action='store_true', help='Process all series regardless of orientation')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug output')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of studies to process')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model based on model_type
    if args.model_type.lower() == 'resnet':
        print(f"Using ResNetSpineModel with ResNet18 backbone")
        model = ResNetSpineModel(in_channels=1, dropout_rate=0.0, pretrained=False)  # No dropout for inference
    else:
        print(f"Using SimpleSpineModel with minimal backbone")
        model = SimpleSpineModel(in_channels=1, dropout_rate=0.0)  # No dropout for inference
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded {args.model_type} model from {args.model_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get DICOM file stats and check preprocessing
    debug_dir = os.path.join(args.output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Track statistics
    processed_series = 0
    skipped_series = 0
    
    # Process each study in test_dir
    test_studies = sorted([d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))])
    
    # Limit number of studies if specified
    if args.limit and args.limit > 0:
        test_studies = test_studies[:args.limit]
    
    for study_id in tqdm(test_studies, desc="Processing studies"):
        study_dir = os.path.join(args.test_dir, study_id)
        
        # Process each series in study
        test_series = sorted([d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))])
        for series_id in tqdm(test_series, desc=f"Study {study_id} series", leave=False):
            series_dir = os.path.join(study_dir, series_id)
            
            # Check if series is sagittal or if we're forcing processing of all series
            is_sagittal = is_sagittal_series(series_dir, debug=args.debug)
            
            if is_sagittal or args.force_all:
                if is_sagittal:
                    print(f"  Series {series_id} is sagittal, running predictions...")
                else:
                    print(f"  Series {series_id} is not sagittal, but processing anyway due to --force_all...")
                
                # Run predictions with enhanced debugging
                results = predict_series_with_debug(
                    model, 
                    series_dir, 
                    device,
                    args.output_dir,
                    target_size=(args.image_size, args.image_size)
                )
                
                if results:
                    # Create enhanced visualizations
                    visualize_results_with_debug(
                        results, 
                        args.output_dir, 
                        study_id, 
                        series_id
                    )
                    print(f"  Results saved to {args.output_dir}")
                    processed_series += 1
            else:
                print(f"  Series {series_id} is not sagittal, skipping")
                skipped_series += 1
    
    print(f"\nTesting complete! Processed {processed_series} series, skipped {skipped_series} series.")
    
    if processed_series == 0:
        print("\nNo series were processed! Try using the --force_all flag to process all series")
        print("or the --debug flag to see detailed orientation information.")
        print("\nExample: python test_spine_classifier.py --test_dir /path/to/data --model_path ./output/best_model_resnet.pth --model_type resnet --force_all --debug")

if __name__ == "__main__":
    main()