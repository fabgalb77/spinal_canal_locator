#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset for slice classification (determining optimal slices for each spinal level).
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


class SliceClassificationDataset(Dataset):
    """Dataset for slice classification (optimal slice identification)."""
    
    def __init__(
        self,
        data_dir: str,
        canal_slices_file: str,
        mode: str = "train",
        split_ratio: float = 0.8,
        target_size: Tuple[int, int] = (128, 128),
        transform = None,
        seed: int = 42,
        include_negatives: bool = True,
        negative_ratio: float = 3.0,  # Increased from 1.0 to 3.0 for better balance
        hard_negative_mining: bool = True,
        series_sampling: bool = True,
        include_adjacent_slices: bool = True  # NEW: Add adjacent slices as channels
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
            include_negatives: Whether to include negative examples
            negative_ratio: Ratio of negative examples to include
            hard_negative_mining: Whether to use hard negative mining
            series_sampling: Sample from same series for improved discrimination
            include_adjacent_slices: Whether to include adjacent slices as channels
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        self.transform = transform
        self.seed = seed
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        self.hard_negative_mining = hard_negative_mining
        self.series_sampling = series_sampling
        self.include_adjacent_slices = include_adjacent_slices
        
        # Define level names and indices
        self.level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        self.level_to_idx = {name: i for i, name in enumerate(self.level_names)}
        
        # Load canal slices
        with open(canal_slices_file, 'r') as f:
            self.canal_slices = json.load(f)
        
        # Map series to all their available slices (for adjacent slice lookup)
        self.series_to_slices = self._map_series_to_slices()
        
        # Create dataset samples
        self.samples = self._create_samples()
        
        # Split dataset
        self._split_dataset(split_ratio)
        
        # Set default transform if None
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _map_series_to_slices(self) -> Dict:
        """
        Map each series to all its available slices (for adjacent slice lookup).
        
        Returns:
            Dictionary mapping (study_id, series_id) to sorted list of instance numbers
        """
        series_to_slices = {}
        
        for study_id, study_data in self.canal_slices.items():
            for series_id, series_data in study_data.items():
                # Get all instances by scanning directory
                series_dir = os.path.join(self.data_dir, study_id, series_id)
                if os.path.exists(series_dir):
                    key = (study_id, series_id)
                    series_to_slices[key] = []
                    
                    dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                    for f in dicom_files:
                        try:
                            instance_num = int(f.split('.')[0])
                            series_to_slices[key].append(instance_num)
                        except:
                            pass
                    
                    # Sort instances by number (important for adjacent slices)
                    series_to_slices[key].sort()
        
        return series_to_slices
    
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
                
                # Process each level
                for level, level_data in series_data.items():
                    if level not in self.level_names:
                        continue
                        
                    level_idx = self.level_to_idx[level]
                    
                    # Track optimal slices for this study/series/level
                    key = (study_id, series_id, level)
                    optimal_instances = set(level_data['instances'])
                    optimal_slices[key] = optimal_instances
                    
                    # Only add optimal instances to positive samples if they exist in the directory
                    for idx, instance_number in enumerate(level_data['instances']):
                        # Check if the instance exists
                        file_path = os.path.join(self.data_dir, study_id, series_id, f"{instance_number}.dcm")
                        if os.path.exists(file_path):
                            # Get coordinates (although not used for classification, helps with debugging)
                            x, y = level_data['coordinates'][idx]
                            
                            # Add to positive samples
                            sample = {
                                'study_id': study_id,
                                'series_id': series_id,
                                'instance_number': instance_number,
                                'level': level,
                                'level_idx': level_idx,
                                'is_optimal': True,
                                'x': x,  # Not used for classification but good for reference
                                'y': y   # Not used for classification but good for reference
                            }
                            positive_samples.append(sample)
        
        # Second pass: create negative examples with hard negative mining if enabled
        if self.include_negatives:
            for (study_id, series_id, level), optimal_instances in optimal_slices.items():
                level_idx = self.level_to_idx[level]
                
                # Get all instances for this series
                series_key = (study_id, series_id)
                if series_key not in self.series_to_slices:
                    continue
                    
                all_instances = self.series_to_slices[series_key]
                
                # Filter out instances that don't exist in the directory
                available_instances = []
                for inst in all_instances:
                    file_path = os.path.join(self.data_dir, study_id, series_id, f"{inst}.dcm")
                    if os.path.exists(file_path):
                        available_instances.append(inst)
                
                # Only proceed if we have both optimal and non-optimal instances
                if optimal_instances and available_instances:
                    # Calculate number of negative samples to include
                    if self.hard_negative_mining:
                        # Hard negative mining: prioritize instances close to optimal ones
                        # Sort available instances by how close they are to optimal ones
                        sorted_instances = sorted(
                            available_instances,
                            key=lambda x: min(abs(x - opt) for opt in optimal_instances)
                        )
                        
                        # Take instances close to optimal ones first (hard negatives)
                        non_optimal = [i for i in sorted_instances if i not in optimal_instances]
                        
                        # Determine how many negatives to include
                        num_optimal = len(optimal_instances)
                        num_to_include = int(num_optimal * self.negative_ratio)
                        num_to_include = min(num_to_include, len(non_optimal))
                        
                        # Get the closest non-optimal instances (hard negatives)
                        selected_instances = non_optimal[:num_to_include]
                    else:
                        # Simple random sampling of negative examples
                        non_optimal = [i for i in available_instances if i not in optimal_instances]
                        
                        # Determine how many negatives to include
                        num_optimal = len(optimal_instances)
                        num_to_include = int(num_optimal * self.negative_ratio)
                        num_to_include = min(num_to_include, len(non_optimal))
                        
                        # Random selection of non-optimal instances
                        random.seed(self.seed)  # for reproducibility
                        selected_instances = random.sample(non_optimal, num_to_include)
                    
                    # Create negative samples
                    for instance_number in selected_instances:
                        sample = {
                            'study_id': study_id,
                            'series_id': series_id,
                            'instance_number': instance_number,
                            'level': level,
                            'level_idx': level_idx,
                            'is_optimal': False,
                            'x': -1,  # No valid coordinates for negative examples
                            'y': -1
                        }
                        negative_samples.append(sample)
        
        # Combine positive and negative samples
        all_samples = positive_samples + negative_samples
        
        # Shuffle samples
        random.seed(self.seed)
        random.shuffle(all_samples)
        
        # Log the balance
        positive_count = len(positive_samples)
        negative_count = len(negative_samples)
        total = positive_count + negative_count
        if total > 0:
            negative_percent = (negative_count / total) * 100
        else:
            negative_percent = 0
        print(f"Classification Dataset balance: {positive_count} positive, {negative_count} negative "
              f"({negative_percent:.2f}% negative)")
        
        return all_samples
    
    def _find_adjacent_slices(self, study_id, series_id, instance_number):
        """
        Find adjacent slices to the given instance.
        
        Args:
            study_id: Study ID
            series_id: Series ID
            instance_number: Instance number
            
        Returns:
            Tuple of (previous_instance, next_instance) or (None, None) if not found
        """
        series_key = (study_id, series_id)
        if series_key not in self.series_to_slices:
            return None, None
            
        instances = self.series_to_slices[series_key]
        if not instances:
            return None, None
            
        # Find index of current instance in the sorted list
        try:
            idx = instances.index(instance_number)
        except ValueError:
            return None, None
        
        # Get previous and next instances if available
        prev_instance = instances[idx - 1] if idx > 0 else None
        next_instance = instances[idx + 1] if idx < len(instances) - 1 else None
        
        return prev_instance, next_instance
    
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
        
        # Log the split information
        print(f"Split studies: {len(train_studies)} train, {len(val_studies)} val, {len(test_studies)} test")
        
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
                A.GridDistortion(p=0.3),  # Add more augmentations for classification
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Add blur augmentation
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
                return np.zeros((self.target_size[0], self.target_size[1], 1), dtype=np.float32)
            
            # Read DICOM
            dicom = pydicom.dcmread(file_path)
            
            # Get pixel array
            image = dicom.pixel_array.astype(np.float32)
            
            # Normalize to [0, 1]
            if image.max() > 0:
                image = image / image.max()
            
            # Reshape to have channel dimension (single channel)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {study_id}/{series_id}/{instance_number}: {e}")
            # Return a blank image
            return np.zeros((self.target_size[0], self.target_size[1], 1), dtype=np.float32)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with image and classification target
        """
        # Get sample
        sample = self.samples[idx]
        study_id = sample['study_id']
        series_id = sample['series_id']
        instance_number = sample['instance_number']
        
        # Load current slice
        current_slice = self._load_dicom(study_id, series_id, instance_number)
        
        # Get original image dimensions
        orig_height, orig_width = current_slice.shape[:2]
        
        if self.include_adjacent_slices:
            # Find adjacent slices
            prev_instance, next_instance = self._find_adjacent_slices(study_id, series_id, instance_number)
            
            # Load adjacent slices
            prev_slice = self._load_dicom(study_id, series_id, prev_instance) if prev_instance else np.zeros_like(current_slice)
            next_slice = self._load_dicom(study_id, series_id, next_instance) if next_instance else np.zeros_like(current_slice)
            
            # Create multi-channel image with adjacent slices
            # Use first channel of each slice and stack them
            multi_slice = np.concatenate([
                prev_slice[:, :, 0:1],
                current_slice[:, :, 0:1],
                next_slice[:, :, 0:1]
            ], axis=2)
            
            # Apply transforms to multi-channel image
            if self.transform:
                transformed = self.transform(image=multi_slice)
                image = transformed['image']
            else:
                # Manual conversion to tensor
                image = torch.tensor(multi_slice.transpose(2, 0, 1), dtype=torch.float32)
        else:
            # Just use current slice as 3-channel image by duplicating
            rgb_slice = np.concatenate([current_slice] * 3, axis=2)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=rgb_slice)
                image = transformed['image']
            else:
                # Manual conversion to tensor
                image = torch.tensor(rgb_slice.transpose(2, 0, 1), dtype=torch.float32)
        
        # Create target tensor (1 if optimal, 0 if not)
        is_optimal = torch.tensor([float(sample['is_optimal'])], dtype=torch.float32)
        
        # Return dictionary
        return {
            'image': image,
            'is_optimal': is_optimal,
            'level_idx': sample['level_idx'],
            'level': sample['level'],
            'study_id': sample['study_id'],
            'series_id': sample['series_id'],
            'instance_number': sample['instance_number'],
            'coords': (sample['x'], sample['y']),  # For reference only
            'orig_dimensions': (orig_width, orig_height)
        }


class SliceClassificationDataModule:
    """Data module for slice classification."""
    
    def __init__(
        self,
        data_dir: str,
        canal_slices_file: str,
        batch_size: int = 16,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (128, 128),
        split_ratio: float = 0.8,
        seed: int = 42,
        include_negatives: bool = True,
        negative_ratio: float = 3.0,  # Increased ratio
        hard_negative_mining: bool = True,
        series_sampling: bool = True,
        include_adjacent_slices: bool = True  # NEW: Add adjacent slices as channels
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
            hard_negative_mining: Whether to use hard negative mining
            series_sampling: Sample from same series for improved discrimination
            include_adjacent_slices: Whether to include adjacent slices as channels
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
        self.hard_negative_mining = hard_negative_mining
        self.series_sampling = series_sampling
        self.include_adjacent_slices = include_adjacent_slices
        
        # Training transforms with more augmentations for classification
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GridDistortion(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Add blur augmentation
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
        self.train_dataset = SliceClassificationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="train",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.train_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio,
            hard_negative_mining=self.hard_negative_mining,
            series_sampling=self.series_sampling,
            include_adjacent_slices=self.include_adjacent_slices
        )
        
        self.val_dataset = SliceClassificationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="val",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio,
            hard_negative_mining=self.hard_negative_mining,
            series_sampling=self.series_sampling,
            include_adjacent_slices=self.include_adjacent_slices
        )
        
        self.test_dataset = SliceClassificationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="test",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio,
            hard_negative_mining=self.hard_negative_mining,
            series_sampling=self.series_sampling,
            include_adjacent_slices=self.include_adjacent_slices
        )
        
        # Print dataset statistics
        print(f"Classification Train dataset: {len(self.train_dataset)} samples")
        print(f"Classification Validation dataset: {len(self.val_dataset)} samples")
        print(f"Classification Test dataset: {len(self.test_dataset)} samples")
    
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