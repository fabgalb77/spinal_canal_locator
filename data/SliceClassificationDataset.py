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
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SliceClassificationDataset(Dataset):
    """
    Dataset for spine slice level classification using a single-slice approach.
    No adjacent slices are used to prevent position bias.
    """
    
    def __init__(
        self,
        data_dir: str,
        canal_slices: Dict,
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (128, 128),
        mode: str = "train",
        include_negatives: bool = True,
        negative_ratio: float = 3.0,
        hard_negative_mining: bool = False
    ):
        """Initialize dataset."""
        self.data_dir = data_dir
        self.canal_slices = canal_slices
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio
        self.hard_negative_mining = hard_negative_mining
        
        # Create samples
        self.samples = self._create_samples()
    
    def __len__(self):
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get dataset item."""
        # Get sample
        sample = self.samples[idx]
        
        # Get sample data
        study_id = sample["study_id"]
        series_id = sample["series_id"]
        instance_number = sample["instance_number"]
        is_level = sample["is_level"]
        level_idx = sample["level_idx"]
        
        # Load slice (SINGLE SLICE APPROACH - no adjacent slices)
        slice_data = self._load_slice(study_id, series_id, instance_number)
        
        # Extract single channel
        single_channel = slice_data[:, :, 0:1]  # Extract first channel
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=single_channel)
            single_channel = transformed["image"]
        
        # Convert to tensor
        slice_tensor = torch.from_numpy(single_channel.transpose(2, 0, 1)).float()
        
        # Return data
        return {
            "image": slice_tensor,
            "label": float(is_level),
            "level_idx": level_idx,
            "metadata": {
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": instance_number
            }
        }
    
    def _load_slice(self, study_id, series_id, instance_number):
        """Load single slice image."""
        # Construct file path
        file_path = os.path.join(
            self.data_dir,
            str(study_id),
            str(series_id),
            f"{instance_number}.jpg"
        )
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image {file_path}")
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        return image

    
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
        
        if self.mode == "train" and self.leave_out_level_locations:
            # Filter out samples that match any leave_out_level_locations
            filtered_samples = []
            for sample in all_samples:
                level = sample['level']
                # Get instance count for this level in this series
                level_key = f"{level}_{self._get_instance_count(sample)}"
                if level_key not in self.leave_out_level_locations:
                    filtered_samples.append(sample)
            all_samples = filtered_samples

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

    def _get_instance_count(self, sample):
        """Get the count of instances for this sample's level/series."""
        study_id = sample['study_id']
        series_id = sample['series_id']
        level = sample['level']
        
        study_data = self.canal_slices.get(study_id, {})
        series_data = study_data.get(series_id, {})
        level_data = series_data.get(level, {})
        
        return len(level_data.get('instances', []))
    
    
    
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
    
    

class SliceClassificationDataModule:
    """Data module for slice classification with position-agnostic approach."""
    
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
        negative_ratio: float = 3.0,
        hard_negative_mining: bool = True,
        series_sampling: bool = True
    ):
        """Initialize data module."""
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
        
        # Load canal slices
        with open(canal_slices_file, "r") as f:
            self.canal_slices = json.load(f)
        
        # Set up datasets and dataloaders
        self.setup()
    
    def setup(self):
        """Set up datasets and dataloaders."""
        # Create train/validation split
        all_studies = list(self.canal_slices.keys())
        random.seed(self.seed)
        random.shuffle(all_studies)
        
        split_idx = int(len(all_studies) * self.split_ratio)
        train_studies = set(all_studies[:split_idx])
        val_studies = set(all_studies[split_idx:])
        
        # Create strong augmentation for training
        train_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.target_size[0],
                width=self.target_size[1],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ])
        
        # Create validation transform
        val_transform = A.Compose([
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1]
            ),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ])
        
        # Create datasets
        self.train_dataset = SliceClassificationDataset(
            data_dir=self.data_dir,
            canal_slices=self.canal_slices,
            transform=train_transform,
            target_size=self.target_size,
            mode="train",
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio,
            hard_negative_mining=self.hard_negative_mining,
            series_level_selection=self.series_sampling
        )
        
        self.val_dataset = SliceClassificationDataset(
            data_dir=self.data_dir,
            canal_slices=self.canal_slices,
            transform=val_transform,
            target_size=self.target_size,
            mode="val",
            include_negatives=self.include_negatives,
            negative_ratio=self.negative_ratio,
            hard_negative_mining=False,
            series_level_selection=self.series_sampling
        )
        
        # Filter datasets by study
        self.train_dataset.samples = [s for s in self.train_dataset.samples if s["study_id"] in train_studies]
        self.val_dataset.samples = [s for s in self.val_dataset.samples if s["study_id"] in val_studies]
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def train_dataloader(self):
        """Get train dataloader."""
        return self.train_dataloader
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return self.val_dataloader