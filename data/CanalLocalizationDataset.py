#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset for spinal canal localization (coordinates prediction).
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


class CanalLocalizationDataset(Dataset):
    """Dataset for spinal canal localization - focused on coordinate prediction."""
    
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
        only_positive_samples: bool = True  # Only include slices with annotations
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
            only_positive_samples: Only include slices with annotations
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        self.transform = transform
        self.seed = seed
        self.sigma = sigma
        self.only_positive_samples = only_positive_samples
        
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
        Create dataset samples, focusing only on slices with annotations.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Process each study/series/level combination
        for study_id, study_data in self.canal_slices.items():
            for series_id, series_data in study_data.items():
                for level, level_data in series_data.items():
                    if level not in self.level_names:
                        continue
                        
                    level_idx = self.level_to_idx[level]
                    
                    # Create positive samples from annotated slices
                    for idx, instance_number in enumerate(level_data['instances']):
                        # Check if file exists
                        file_path = os.path.join(
                            self.data_dir, 
                            study_id, 
                            series_id, 
                            f"{instance_number}.dcm"
                        )
                        
                        if not os.path.exists(file_path):
                            continue
                            
                        # Get coordinates
                        x, y = level_data['coordinates'][idx]
                        
                        # Create sample
                        sample = {
                            'study_id': study_id,
                            'series_id': series_id,
                            'instance_number': instance_number,
                            'level': level,
                            'level_idx': level_idx,
                            'x': x,
                            'y': y
                        }
                        samples.append(sample)
        
        # Shuffle samples
        random.seed(self.seed)
        random.shuffle(samples)
        
        # Log the sample count
        print(f"Localization Dataset: {len(samples)} samples")
        
        # Log level distribution
        level_counts = {}
        for sample in samples:
            level = sample['level']
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
        
        for level, count in level_counts.items():
            print(f"  Level {level}: {count} samples")
        
        return samples
    
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
            
            # Normalize to [0, 1] properly handling negative values
            if image.max() > image.min():  # Check there's some variation in the image
                image = (image - image.min()) / (image.max() - image.min())
            else:
                # If image is constant (max = min), set to 0.5
                image = np.ones_like(image) * 0.5
            
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
            Dictionary with image and target heatmap
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
        
        # Get coordinates
        orig_x, orig_y = sample['x'], sample['y']
        
        # Create heatmap at original resolution
        heatmap = self._create_heatmap(
            image_shape=(orig_height, orig_width),
            x=orig_x,
            y=orig_y
        )
        
        # Apply transforms to both image and heatmap
        # Also transform keypoints if using coordinate-based transforms
        keypoints = [(orig_x, orig_y)]
        
        if self.transform:
            transformed = self.transform(
                image=image, 
                mask=heatmap,
                keypoints=keypoints
            )
            image = transformed['image']
            heatmap = transformed['mask']
            
            # Get transformed coordinates if available
            if 'keypoints' in transformed and transformed['keypoints']:
                scaled_x, scaled_y = transformed['keypoints'][0]
            else:
                # Calculate scaled coordinates manually
                x_scale = self.target_size[1] / orig_width
                y_scale = self.target_size[0] / orig_height
                scaled_x = orig_x * x_scale
                scaled_y = orig_y * y_scale
        else:
            # Manual conversion to tensor
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            
            # Calculate scaled coordinates
            x_scale = self.target_size[1] / orig_width
            y_scale = self.target_size[0] / orig_height
            scaled_x = orig_x * x_scale
            scaled_y = orig_y * y_scale
        
        # Return dictionary
        return {
            'image': image,
            'heatmap': heatmap.unsqueeze(0),  # Add channel dimension [1, H, W]
            'level_idx': sample['level_idx'],
            'level': sample['level'],
            'study_id': sample['study_id'],
            'series_id': sample['series_id'],
            'instance_number': sample['instance_number'],
            'orig_coordinates': (orig_x, orig_y),
            'orig_dimensions': (orig_width, orig_height),
            'scaled_coordinates': (scaled_x, scaled_y)
        }


class CanalLocalizationDataModule:
    """Data module for spinal canal localization."""
    
    def __init__(
        self,
        data_dir: str,
        canal_slices_file: str,
        batch_size: int = 8,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (128, 128),
        split_ratio: float = 0.8,
        seed: int = 42,
        only_positive_samples: bool = True
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
            only_positive_samples: Only include slices with annotations
        """
        self.data_dir = data_dir
        self.canal_slices_file = canal_slices_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.only_positive_samples = only_positive_samples
        
        # Updated Training transforms
        """
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            # Replace ShiftScaleRotate with Affine
            A.Affine(
                scale=(0.85, 1.15),       # Scale between 85% and 115%
                translate_percent=(0.2, 0.2),  # Translation up to 20% in either direction
                rotate=(-20, 20),         # Rotation up to 20 degrees
                shear=(-5, 5),            # Add slight shearing
                border_mode=cv2.BORDER_CONSTANT,
                p=0.9
            ),
            
            # Fix ElasticTransform parameters
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=0.5
            ),
            
            # Keep GridDistortion as is
            A.GridDistortion(
                distort_limit=0.1,
                p=0.5
            ),
            
            # Keep RandomBrightnessContrast as is
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.7
            ),
            
            A.Blur(blur_limit=3, p=0.4),
            
            # Keep RandomGamma as is
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5
            ),
            
            # Standard normalization
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        """
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # Validation transforms
        self.val_transform = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    
    def setup(self):
        """Set up datasets."""
        # Create datasets
        self.train_dataset = CanalLocalizationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="train",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.train_transform,
            seed=self.seed,
            only_positive_samples=self.only_positive_samples
        )
        
        self.val_dataset = CanalLocalizationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="val",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            only_positive_samples=self.only_positive_samples
        )
        
        self.test_dataset = CanalLocalizationDataset(
            data_dir=self.data_dir,
            canal_slices_file=self.canal_slices_file,
            mode="test",
            split_ratio=self.split_ratio,
            target_size=self.target_size,
            transform=self.val_transform,
            seed=self.seed,
            only_positive_samples=self.only_positive_samples
        )
        
        # Print dataset statistics
        print(f"Localization Train dataset: {len(self.train_dataset)} samples")
        print(f"Localization Validation dataset: {len(self.val_dataset)} samples")
        print(f"Localization Test dataset: {len(self.test_dataset)} samples")
    
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
