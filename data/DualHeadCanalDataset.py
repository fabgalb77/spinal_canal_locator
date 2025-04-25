#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset for dual-headed level-specific spinal canal localization.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import json
import cv2
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DualHeadCanalDataset(Dataset):
    """Dataset for dual-headed level-specific spinal canal localization."""
    
    def __init__(
        self,
        data_dir: str,
        canal_slices_file: str,
        mode: str = "train",
        split_ratio: float = 0.8,
        target_size: Tuple[int, int] = (128, 128),
        transform = None,
        seed: int = 42,
        sigma: float = 5.0,  # Gaussian sigma for heatmaps
        include_negatives: bool = True,  # Whether to include negative examples
        negative_ratio: float = 0.2  # Ratio of negative examples to include
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory for DICOM files
            canal_slices_file: JSON file with canal slices information
            mode: "train", "val", or "test"
            split_ratio: Ratio of training data
            target_size: Target image size
            transform: Albumentations transforms
            seed: Random seed
            sigma: Sigma for Gaussian heatmaps
            include_negatives: Whether to include negative examples
            negative_ratio: Ratio of negative examples to include
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        self.transform = transform
        self.seed = seed
        self.sigma = sigma
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        
        # Define level names and indices
        self.level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        self.level_to_idx = {name: i for i, name in enumerate(self.level_names)}
        
        # Load canal slices
        with open(canal_slices_file, 'r') as f:
            self.canal_slices = json.load(f)
        
        # Create dataset samples
        self.samples = self._create_samples()
        
        # Split dataset
        self._split_dataset(split_ratio)
        
        # Set default transform if None
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _create_samples(self) -> List[Dict]:
        """
        Create dataset samples with proper balance of positive and negative examples.
        
        Positive samples: Slices that are annotated as the optimal view for a specific level
        Negative samples: All other slices that aren't the optimal view for that level
        
        The negative_ratio parameter controls how many negative samples to include
        relative to the number of positive samples.
        
        Returns:
            List of sample dictionaries
        """
        positive_samples = []
        negative_samples = []
        
        # Track which slices are optimal for which levels
        optimal_slices = {}  # (study_id, series_id, level) -> set of optimal instance numbers
        
        # First pass: collect all optimal (annotated) slices and create positive samples
        for study_id, study_data in self.canal_slices.items():
            for series_id, series_data in study_data.items():
                for level, level_data in series_data.items():
                    if level not in self.level_names:
                        continue
                        
                    level_idx = self.level_to_idx[level]
                    
                    # Track optimal slices for this study/series/level
                    key = (study_id, series_id, level)
                    optimal_slices[key] = set(level_data['instances'])
                    
                    # Create positive samples from annotated slices
                    for idx, instance_number in enumerate(level_data['instances']):
                        x, y = level_data['coordinates'][idx]
                        sample = {
                            'study_id': study_id,
                            'series_id': series_id,
                            'instance_number': instance_number,
                            'level': level,
                            'level_idx': level_idx,
                            'x': x,
                            'y': y,
                            'level_present': 1.0,  # Optimal view
                            'is_optimal': True
                        }
                        positive_samples.append(sample)
        
        # Second pass: find all slices in each series to create negative samples
        for study_id, study_data in self.canal_slices.items():
            for series_id, series_data in study_data.items():
                # Get a list of all levels in this series
                levels_in_series = [l for l in series_data.keys() if l in self.level_names]
                
                # Get a list of all instance numbers in this series
                all_instances = set()
                
                # Try to get instances by scanning directory first (more comprehensive)
                series_dir = os.path.join(self.data_dir, study_id, series_id)
                if os.path.exists(series_dir):
                    dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                    for f in dicom_files:
                        try:
                            instance_num = int(f.split('.')[0])
                            all_instances.add(instance_num)
                        except:
                            pass
                
                # If directory scan failed or found no instances, fall back to annotation data
                if not all_instances:
                    for level, level_data in series_data.items():
                        if level in self.level_names:
                            all_instances.update(level_data['instances'])
                
                # Create negative samples for each level
                for level in levels_in_series:
                    level_idx = self.level_to_idx[level]
                    optimal_insts = optimal_slices.get((study_id, series_id, level), set())
                    
                    # For each instance that is not optimal for this level, create a negative sample
                    for inst in all_instances:
                        if inst not in optimal_insts:
                            sample = {
                                'study_id': study_id,
                                'series_id': series_id,
                                'instance_number': inst,
                                'level': level,
                                'level_idx': level_idx,
                                'x': -1,  # No valid coordinates
                                'y': -1,
                                'level_present': 0.0,  # Not optimal
                                'is_optimal': False
                            }
                            negative_samples.append(sample)
        
        # Apply negative_ratio to control the balance between positive and negative examples
        if self.negative_ratio > 0:
            # Calculate how many negative samples to keep based on ratio
            target_neg_count = int(len(positive_samples) * self.negative_ratio)
            
            # If we have more negatives than the target, randomly subsample
            if len(negative_samples) > target_neg_count:
                # Set random seed for reproducibility
                random.seed(self.seed)
                negative_samples = random.sample(negative_samples, target_neg_count)
                
            # Log the subsampling
            original_count = len(negative_samples)
            print(f"Subsampled negative examples: {len(negative_samples)}/{original_count} " 
                  f"(target ratio: {self.negative_ratio})")
        
        # Combine positive and negative samples
        all_samples = positive_samples + negative_samples
        
        # Shuffle samples
        random.seed(self.seed)
        random.shuffle(all_samples)
        
        # Log the balance
        positive_count = len(positive_samples)
        negative_count = len(negative_samples)
        print(f"Dataset balance: {positive_count} positive, {negative_count} negative "
              f"({negative_count/(positive_count+negative_count):.2%} negative)")
        
        return all_samples
    
    def _split_dataset(self, split_ratio: float) -> None:
        """
        Split dataset into train/val/test.
        
        Args:
            split_ratio: Ratio of training data
        """
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Get all unique study IDs
        study_ids = list(set(sample['study_id'] for sample in self.samples))
        
        # Shuffle study IDs
        random.shuffle(study_ids)
        
        # Split into train and val/test
        train_size = int(len(study_ids) * split_ratio)
        train_studies = set(study_ids[:train_size])
        val_test_studies = set(study_ids[train_size:])
        
        # Further split val/test if needed
        val_size = len(val_test_studies) // 2
        val_studies = set(list(val_test_studies)[:val_size])
        test_studies = set(list(val_test_studies)[val_size:])
        
        # Filter samples based on mode
        if self.mode == "train":
            self.samples = [s for s in self.samples if s['study_id'] in train_studies]
        elif self.mode == "val":
            self.samples = [s for s in self.samples if s['study_id'] in val_studies]
        elif self.mode == "test":
            self.samples = [s for s in self.samples if s['study_id'] in test_studies]
        else:  # Use all samples
            pass
    
    def _get_default_transforms(self):
        """
        Get default data augmentation transforms.
        
        Returns:
            Albumentations transforms
        """
        if self.mode == "train":
            return A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
    
    def _load_dicom(self, study_id: str, series_id: str, instance_number: int) -> np.ndarray:
        """
        Load a DICOM image.
        
        Args:
            study_id: Study ID
            series_id: Series ID
            instance_number: Instance number
            
        Returns:
            Image as numpy array
        """
        try:
            # Construct path
            file_path = os.path.join(
                self.data_dir, 
                study_id, 
                series_id, 
                f"{instance_number}.dcm"
            )
            
            # Check if file exists
            if not os.path.exists(file_path):
                # Return a blank image
                return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            
            # Read DICOM
            dicom = pydicom.dcmread(file_path)
            
            # Get pixel array
            image = dicom.pixel_array.astype(np.float32)
            
            # Normalize to [0, 1]
            if image.max() > 0:
                image = image / image.max()
            
            # Convert to RGB (3 channels)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {study_id}/{series_id}/{instance_number}: {e}")
            # Return a blank image
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
    
    def _create_heatmap(self, image_shape: Tuple[int, int], x: float, y: float) -> np.ndarray:
        """
        Create a Gaussian heatmap centered at (x, y).
        
        Args:
            image_shape: Shape of the image (height, width)
            x: X-coordinate in original space
            y: Y-coordinate in original space
            
        Returns:
            Heatmap as numpy array with same shape as image
        """
        # Get image dimensions - this is the original image size
        height, width = image_shape
        
        # Create coordinate grids for the original image dimensions
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        
        # Create Gaussian heatmap at original image dimensions
        heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * self.sigma ** 2))
        
        # Ensure the heatmap has exactly the same shape as the image
        assert heatmap.shape == (height, width), f"Heatmap shape {heatmap.shape} doesn't match image shape {(height, width)}"
        
        return heatmap
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with image and target heatmaps
        """
        # Get sample
        sample = self.samples[idx]
        
        # Load image
        image = self._load_dicom(
            sample['study_id'],
            sample['series_id'],
            sample['instance_number']
        )
        
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # For positive examples, create a heatmap
        if sample['level_present'] > 0.5:  # Positive example
            # Create heatmap at original resolution
            heatmap = self._create_heatmap(
                image_shape=(orig_height, orig_width),
                x=sample['x'],
                y=sample['y']
            )
            
            # Store original coordinates
            orig_x, orig_y = sample['x'], sample['y']
        else:  # Negative example
            # Create a blank heatmap
            heatmap = np.zeros((orig_height, orig_width), dtype=np.float32)
            
            # No valid coordinates for negative examples
            orig_x, orig_y = -1, -1
        
        # Apply transforms to both image and heatmap
        if self.transform:
            transformed = self.transform(image=image, mask=heatmap)
            image = transformed['image']
            heatmap = transformed['mask']
        else:
            # Manual conversion to tensor
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
        
        # Calculate scaled coordinates for the target size (for positive examples only)
        if sample['level_present'] > 0.5:
            x_scale = self.target_size[1] / orig_width
            y_scale = self.target_size[0] / orig_height
            
            scaled_x = orig_x * x_scale
            scaled_y = orig_y * y_scale
        else:
            scaled_x, scaled_y = -1, -1
        
        # Return dictionary
        return {
            'image': image,
            'heatmap': heatmap,
            'level_idx': sample['level_idx'],
            'level': sample['level'],
            'level_present': torch.tensor([sample['level_present']], dtype=torch.float32),
            'study_id': sample['study_id'],
            'series_id': sample['series_id'],
            'instance_number': sample['instance_number'],
            'orig_coordinates': (orig_x, orig_y),
            'orig_dimensions': (orig_width, orig_height),
            'scaled_coordinates': (scaled_x, scaled_y)
        }


class DualHeadCanalDataModule:
    """Data module for dual-headed level-specific spinal canal localization."""
    
    def __init__(
        self,
        data_dir: str,
        canal_slices_file: str,
        batch_size: int = 8,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (128, 128),
        split_ratio: float = 0.8,
        seed: int = 42,
        include_negatives: bool = True,
        negative_ratio: float = 0.2
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Root directory for DICOM files
            canal_slices_file: JSON file with canal slices information
            batch_size: Batch size
            num_workers: Number of workers
            target_size: Target image size
            split_ratio: Ratio of training data
            seed: Random seed
            include_negatives: Whether to include negative examples
            negative_ratio: Ratio of negative examples to include
        """
        self.data_dir = data_dir
        self.canal_slices_file = canal_slices_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        
        # Training transforms
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
        # Validation transforms
        self.val_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    
    def setup(self):
        """Set up datasets."""
        # Create datasets
        self.train_dataset = DualHeadCanalDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="train",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.train_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio
        )
        
        self.val_dataset = DualHeadCanalDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="val",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio
        )
        
        self.test_dataset = DualHeadCanalDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="test",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio
        )
        
        # Print dataset statistics
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
        # Print level distribution in training set
        level_counts = {level: {'positive': 0, 'negative': 0} 
                       for level in self.train_dataset.level_names}
        
        for sample in self.train_dataset.samples:
            level = sample['level']
            if sample['level_present'] > 0.5:
                level_counts[level]['positive'] += 1
            else:
                level_counts[level]['negative'] += 1
        
        print("Training set level distribution:")
        for level, counts in level_counts.items():
            print(f"  {level}: {counts['positive']} positive, {counts['negative']} negative samples")
    
    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
