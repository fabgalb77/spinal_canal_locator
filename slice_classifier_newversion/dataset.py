"""
Dataset classes for spine classification.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVEL_TO_IDX = {level: i for i, level in enumerate(LEVEL_NAMES)}

# Configure logging
logger = logging.getLogger(__name__)

def process_dicom(dicom_path, target_size=(256, 256), return_pixel_array=False):
    """
    Standardized DICOM processing function to ensure consistency between
    training and testing pipelines.
    
    Parameters:
    -----------
    dicom_path : str
        Path to the DICOM file
    target_size : tuple
        Target size for the image (height, width)
    return_pixel_array : bool
        Whether to return the raw pixel array for visualization
    
    Returns:
    --------
    dict
        Dictionary containing processed image and metadata
    """
    try:
        # Load DICOM
        dicom = pydicom.dcmread(dicom_path)
        
        # Extract raw pixel array
        pixel_array = dicom.pixel_array
        
        # Simple min-max normalization to 0-255 range
        normalized = pixel_array.copy()
        normalized = normalized - np.min(normalized)
        if np.max(normalized) > 0:
            normalized = normalized / np.max(normalized) * 255
        normalized = normalized.astype(np.uint8)
        
        # Keep a copy of the normalized array for visualization
        viz_array = normalized.copy()
        
        # Ensure single channel (grayscale)
        if len(normalized.shape) > 2:
            normalized = normalized[:, :, 0]
        
        # Resize to target size
        resized = cv2.resize(normalized, target_size)
        
        # Return as a dictionary
        result = {
            "image": resized,
            "instance_number": int(os.path.basename(dicom_path).split('.')[0]),
            "dicom_path": dicom_path,
        }
        
        # Add pixel array for visualization if requested
        if return_pixel_array:
            result["pixel_array"] = viz_array
            
        return result
    
    except Exception as e:
        logger.error(f"Error processing {dicom_path}: {e}")
        # Return empty image in case of error
        result = {
            "image": np.zeros(target_size, dtype=np.uint8),
            "instance_number": -1 if not os.path.basename(dicom_path).split('.')[0].isdigit() else int(os.path.basename(dicom_path).split('.')[0]),
            "dicom_path": dicom_path,
        }
        
        if return_pixel_array:
            result["pixel_array"] = np.zeros(target_size, dtype=np.uint8)
            
        return result

def create_data_transforms(target_size=(256, 256), normalization=True):
    """
    Create standardized data transforms for training and validation.
    
    Parameters:
    -----------
    target_size : tuple
        Target size for the image (height, width)
    normalization : bool
        Whether to include normalization in transforms
    
    Returns:
    --------
    tuple
        Tuple of (train_transform, val_transform)
    """
    train_transforms = []
    val_transforms = []
    
    # Always include resize
    train_transforms.append(A.Resize(height=target_size[0], width=target_size[1]))
    val_transforms.append(A.Resize(height=target_size[0], width=target_size[1]))
    
    # Add training augmentations
    train_transforms.extend([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])
    
    # Add normalization and conversion to tensor for both
    if normalization:
        train_transforms.append(A.Normalize(mean=[0.5], std=[0.5]))
        val_transforms.append(A.Normalize(mean=[0.5], std=[0.5]))
    
    # Add ToTensorV2 as final step for both
    train_transforms.append(ToTensorV2())
    val_transforms.append(ToTensorV2())
    
    train_transform = A.Compose(train_transforms)
    val_transform = A.Compose(val_transforms)
    
    return train_transform, val_transform


class SliceClassificationDataset(Dataset):
    """
    Dataset for spine slice classification using a single-slice approach.
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations: pd.DataFrame = None,
        series: pd.DataFrame = None,
        annotations_file: str = None,
        series_file: str = None,
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        condition: str = "Spinal Canal Stenosis",
        negative_ratio: float = 2.0,
        preloaded_samples: Optional[List[Dict]] = None
    ):
        """Initialize dataset with single-slice approach."""
        self.data_dir = data_dir
        self.target_size = target_size
        self.transform = transform
        self.mode = mode
        self.condition = condition
        self.negative_ratio = negative_ratio
        
        # If samples are preloaded, use them directly
        if preloaded_samples is not None:
            self.samples = preloaded_samples
            positives = sum(1 for sample in self.samples if sample["is_positive"])
            negatives = len(self.samples) - positives
            pos_pct = (positives / len(self.samples)) * 100 if self.samples else 0
            neg_pct = (negatives / len(self.samples)) * 100 if self.samples else 0
            
            logger.info(f"Created {mode} dataset with {len(self.samples)} samples (preloaded)")
            logger.info(f"  Condition: {condition}")
            logger.info(f"  Class balance: {positives} positives ({pos_pct:.1f}%), {negatives} negatives ({neg_pct:.1f}%)")
            return
        
        # Load annotations from file if dataframe not provided
        if annotations is None and annotations_file is not None:
            self.annotations = pd.read_csv(annotations_file)
            self.annotations = self.annotations[self.annotations["condition"] == condition]
        else:
            self.annotations = annotations
        
        # Load series descriptions from file if dataframe not provided
        if series is None and series_file is not None:
            self.series = pd.read_csv(series_file)
            self.sagittal_series = self.series[self.series["series_description"].str.contains("Sagittal", case=False)]
            self.sagittal_ids = set(zip(self.sagittal_series["study_id"], self.sagittal_series["series_id"]))
            
            # Filter annotations to only include sagittal series
            self.annotations = self.annotations[
                self.annotations.apply(lambda row: (row["study_id"], row["series_id"]) in self.sagittal_ids, axis=1)
            ]
        else:
            # Assume provided series dataframe is already filtered for sagittal
            self.sagittal_series = series
        
        # Verify studies exist in data_dir
        self.verified_studies = self._verify_studies_exist()
        logger.info(f"Verified {len(self.verified_studies)} studies exist in {data_dir}")
        
        # Filter annotations to only include verified studies
        initial_count = len(self.annotations)
        self.annotations = self.annotations[self.annotations["study_id"].isin(self.verified_studies)]
        filtered_count = len(self.annotations)
        logger.info(f"Filtered annotations from {initial_count} to {filtered_count} rows (keeping only available studies)")
        
        # Check if any data remains after filtering
        if filtered_count == 0:
            logger.error("No annotations remain after filtering for available studies!")
            self.samples = []
            return
        
        # Get all unique series
        self.unique_series = self.annotations.drop_duplicates(subset=["study_id", "series_id"])[["study_id", "series_id"]]
        logger.info(f"Found {len(self.unique_series)} unique sagittal series with canal stenosis annotations")
        
        # Scan available DICOM files
        self.available_instances = self._scan_available_instances()
        
        # Create samples list with positive and negative examples
        self.samples = self._create_samples()
        
        logger.info(f"Created {mode} dataset with {len(self.samples)} samples")
        logger.info(f"  Condition: {condition}")
        
        # Calculate and log class balance
        positives = sum(1 for sample in self.samples if sample["is_positive"])
        negatives = len(self.samples) - positives
        pos_pct = (positives / len(self.samples)) * 100 if self.samples else 0
        neg_pct = (negatives / len(self.samples)) * 100 if self.samples else 0
        logger.info(f"  Class balance: {positives} positives ({pos_pct:.1f}%), {negatives} negatives ({neg_pct:.1f}%)")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a dataset item with single-slice approach."""
        sample = self.samples[idx]
        
        study_id = sample["study_id"]
        series_id = sample["series_id"]
        instance_number = sample["instance_number"]
        is_positive = sample["is_positive"]
        level_idx = sample["level_idx"]
        
        # Load single slice image from DICOM
        dicom_path = os.path.join(
            self.data_dir, 
            str(study_id), 
            str(series_id), 
            f"{instance_number}.dcm"
        )
        
        # Process the DICOM file using the standardized function
        if os.path.exists(dicom_path):
            result = process_dicom(dicom_path, target_size=self.target_size)
            image = result["image"]
        else:
            # Handle missing file
            image = np.zeros(self.target_size, dtype=np.uint8)
        
        # Make sure image is single channel (H, W, 1)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Convert to tensor (C, H, W)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return {
            "image": image_tensor,
            "label": float(is_positive),
            "level_idx": level_idx,
            "metadata": {
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": instance_number
            }
        }
    
    def _verify_studies_exist(self):
        """Verify which studies in annotations actually exist in data_dir."""
        verified_studies = []
        
        # Get unique study IDs from annotations
        unique_studies = self.annotations["study_id"].unique()
        
        # Check which studies exist in data_dir
        for study_id in unique_studies:
            study_dir = os.path.join(self.data_dir, str(study_id))
            if os.path.exists(study_dir) and os.path.isdir(study_dir):
                verified_studies.append(study_id)
        
        return verified_studies
    
    def _scan_available_instances(self):
        """
        Scan filesystem to find all available DICOM instances for sagittal series.
        Returns a dictionary mapping (study_id, series_id) to lists of instance numbers.
        Only keeps central slices of each series.
        """
        available_instances = {}
        
        # Check for DICOM files for a few series to debug
        sample_size = min(5, len(self.unique_series))
        logger.info(f"Checking DICOM files for {sample_size} example series")
        
        for i in range(sample_size):
            row = self.unique_series.iloc[i]
            study_id = row["study_id"]
            series_id = row["series_id"]
            series_dir = os.path.join(self.data_dir, str(study_id), str(series_id))
            
            logger.info(f"Example {i+1}: {series_dir}, exists: {os.path.exists(series_dir)}")
            
            if os.path.exists(series_dir):
                all_files = os.listdir(series_dir)
                dcm_files = [f for f in all_files if f.endswith('.dcm')]
                logger.info(f"  Found {len(dcm_files)} DICOM files out of {len(all_files)} total files")
                if dcm_files:
                    logger.info(f"  Sample DICOM files: {dcm_files[:5]}")
        
        # Systematically scan all series
        logger.info(f"Scanning for DICOM files in {len(self.unique_series)} series...")
        
        # Get unique study/series combinations from filtered annotations
        for _, row in self.unique_series.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            series_dir = os.path.join(self.data_dir, str(study_id), str(series_id))
            
            if os.path.exists(series_dir):
                # Get all DICOM files in directory
                dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                
                # Extract instance numbers from filenames
                instance_numbers = []
                for file_name in dicom_files:
                    try:
                        # Filename format is "{instance_number}.dcm"
                        instance_number = int(file_name.split('.')[0])
                        instance_numbers.append(instance_number)
                    except ValueError:
                        continue
                
                # Sort instance numbers
                instance_numbers = sorted(instance_numbers)
                
                # Only keep the central slices (5 central slices)
                if len(instance_numbers) > 5:
                    middle_idx = len(instance_numbers) // 2
                    start_idx = max(0, middle_idx - 2)
                    end_idx = min(len(instance_numbers), middle_idx + 3)
                    central_instances = instance_numbers[start_idx:end_idx]
                    logger.info(f"Series {study_id}/{series_id}: Keeping only central slices {central_instances} from {len(instance_numbers)} total slices")
                else:
                    # If there are 5 or fewer slices, keep all of them
                    central_instances = instance_numbers
                    logger.info(f"Series {study_id}/{series_id}: Series has only {len(instance_numbers)} slices, keeping all")
                
                # Store available instances
                if central_instances:
                    available_instances[(study_id, series_id)] = central_instances
        
        # Log statistics
        total_instances = sum(len(instances) for instances in available_instances.values())
        total_series = len(available_instances)
        logger.info(f"Found {total_instances} central DICOM files across {total_series} series")
        
        # Handle case where filesystem scanning fails
        if total_series == 0:
            logger.warning("No DICOM files found! Falling back to synthetic instance ranges")
            return self._create_synthetic_instance_ranges()
        
        return available_instances
    
    def _create_synthetic_instance_ranges(self):
        """
        Create synthetic instance ranges as fallback when filesystem scanning fails.
        Only keeps central slices.
        """
        available_instances = {}
        
        # Group annotations by study_id and series_id
        grouped = self.annotations.groupby(["study_id", "series_id"])
        
        # For each series, create a synthetic range of instance numbers
        for (study_id, series_id), group in grouped:
            # Get actual annotated instances
            actual_instances = group["instance_number"].unique().tolist()
            
            # Calculate range based on min/max
            min_instance = min(actual_instances)
            max_instance = max(actual_instances)
            
            # Create a range that extends beyond the annotated instances
            # Typical sagittal series has 15-25 slices
            extended_range = list(range(max(1, min_instance - 2), max_instance + 3))
            
            # Only keep central slices
            if len(extended_range) > 5:
                middle_idx = len(extended_range) // 2
                start_idx = max(0, middle_idx - 2)
                end_idx = min(len(extended_range), middle_idx + 3)
                central_instances = extended_range[start_idx:end_idx]
                logger.info(f"Series {study_id}/{series_id}: Synthetic range keeping only central slices {central_instances} from {len(extended_range)} total slices")
            else:
                # If there are 5 or fewer slices, keep all of them
                central_instances = extended_range
                logger.info(f"Series {study_id}/{series_id}: Synthetic range has only {len(extended_range)} slices, keeping all")
            
            # Store the central instances
            available_instances[(study_id, series_id)] = central_instances
        
        # Log statistics
        total_instances = sum(len(instances) for instances in available_instances.values())
        logger.info(f"Created synthetic ranges with {total_instances} central instances across {len(available_instances)} series")
        
        return available_instances
    
    def _create_samples(self):
        """Create samples list with positive and negative examples."""
        samples = []
        
        # Group by study, series, level to find instances with annotations
        positive_groups = self.annotations.groupby(["study_id", "series_id", "level"])
        
        # Process each group to create positive and negative samples
        for (study_id, series_id, level), group in positive_groups:
            # Skip if level not in our predefined list
            if level not in LEVEL_TO_IDX:
                continue
            
            level_idx = LEVEL_TO_IDX[level]
            
            # Get the labeled instance(s) for this level
            positive_instances = group["instance_number"].unique()
            
            # Add positive samples
            for instance in positive_instances:
                samples.append({
                    "study_id": study_id,
                    "series_id": series_id,
                    "instance_number": instance,
                    "is_positive": True,
                    "level_idx": level_idx
                })
            
            # Get all available instances for this series
            key = (study_id, series_id)
            if key not in self.available_instances:
                continue  # Skip if we couldn't find instances for this series
                
            all_instances = self.available_instances[key]
            
            # Generate negative samples (non-labeled instances for this level)
            negative_instances = [inst for inst in all_instances if inst not in positive_instances]
            
            # Randomly sample negative instances to maintain ratio
            num_positives = len(positive_instances)
            num_negatives = int(num_positives * self.negative_ratio)
            
            if negative_instances:
                # Sample negative instances (without replacement if possible)
                if num_negatives <= len(negative_instances):
                    sampled_negatives = random.sample(negative_instances, num_negatives)
                else:
                    # With replacement if needed
                    sampled_negatives = random.choices(negative_instances, k=num_negatives)
                
                # Add negative samples
                for instance in sampled_negatives:
                    samples.append({
                        "study_id": study_id,
                        "series_id": series_id,
                        "instance_number": instance,
                        "is_positive": False,
                        "level_idx": level_idx
                    })
        
        return samples


class TestDataset(Dataset):
    """Dataset for testing that matches training preprocessing."""
    
    def __init__(self, series_dir, target_size=(256, 256), transform=None):
        self.series_dir = series_dir
        self.target_size = target_size
        self.transform = transform
        
        # Create default transform if none provided
        if self.transform is None:
            _, self.transform = create_data_transforms(target_size=target_size)
        
        # Get DICOM files
        self.files = sorted([
            os.path.join(series_dir, f) for f in os.listdir(series_dir) 
            if f.endswith('.dcm')
        ], key=lambda x: int(os.path.basename(x).split('.')[0]))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        dicom_path = self.files[idx]
        
        # Process DICOM using the standardized function
        result = process_dicom(dicom_path, target_size=self.target_size, return_pixel_array=True)
        
        # Get processed image
        image = result["image"]
        
        # Make sure image is single channel (H, W, 1)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Convert to tensor (C, H, W)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Debug - print stats if needed
        # print(f"Image tensor stats: min={image_tensor.min().item():.4f}, max={image_tensor.max().item():.4f}, mean={image_tensor.mean().item():.4f}, std={image_tensor.std().item():.4f}")
        
        return {
            "image": image_tensor,
            "instance_number": result["instance_number"],
            "dicom_path": result["dicom_path"],
            "pixel_array": result["pixel_array"],  # Keep original for visualization
        }


def is_sagittal_series(series_dir, debug=True):
    """
    Determine if a series is sagittal based on DICOM tags.
    Checks several indicators including series description, image orientation, 
    and image plane characteristics.
    """
    # Get DICOM files in the directory
    dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
    if not dicom_files:
        if debug:
            print(f"  No DICOM files found in {series_dir}")
        return False

    # Load the first DICOM file to check orientation
    try:
        dicom_path = os.path.join(series_dir, dicom_files[0])
        dicom = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        # Extract potentially useful tags
        series_desc = getattr(dicom, 'SeriesDescription', '').lower()
        protocol_name = getattr(dicom, 'ProtocolName', '').lower() if hasattr(dicom, 'ProtocolName') else ''
        image_type = getattr(dicom, 'ImageType', ['']) if hasattr(dicom, 'ImageType') else ['']
        body_part = getattr(dicom, 'BodyPartExamined', '').lower() if hasattr(dicom, 'BodyPartExamined') else ''
        
        # Debug info
        if debug:
            print(f"  Series {os.path.basename(series_dir)} details:")
            print(f"    Series Description: {series_desc}")
            print(f"    Protocol Name: {protocol_name}")
            print(f"    Body Part: {body_part}")
            print(f"    Image Type: {image_type}")
        
        # Check multiple indicators
        is_spine = 'spine' in series_desc or 'spine' in protocol_name or body_part == 'spine'
        is_sag_in_desc = 'sagittal' in series_desc or 'sag' in series_desc
        is_sag_in_protocol = 'sagittal' in protocol_name or 'sag' in protocol_name
        
        # Check descriptions first - this is most reliable
        if is_sag_in_desc or is_sag_in_protocol:
            if debug:
                print(f"    PASS: Sagittal found in description or protocol")
            return True
            
        # If we have ImageOrientationPatient, check it for sagittal orientation
        if hasattr(dicom, 'ImageOrientationPatient'):
            orientation = dicom.ImageOrientationPatient
            
            if debug:
                print(f"    Image Orientation: {orientation}")
            
            # Check for sagittal orientation based on patient coordinate system
            # Calculate the normal vector to the image plane
            row_x, row_y, row_z = orientation[0:3]
            col_x, col_y, col_z = orientation[3:6]
            
            # The cross product gives us the normal to the image plane
            normal_x = row_y * col_z - row_z * col_y
            normal_y = row_z * col_x - row_x * col_z
            normal_z = row_x * col_y - row_y * col_x
            
            # Normalize
            magnitude = (normal_x**2 + normal_y**2 + normal_z**2)**0.5
            if magnitude > 0:
                normal_x /= magnitude
                normal_y /= magnitude
                normal_z /= magnitude
                
            if debug:
                print(f"    Image Plane Normal: [{normal_x:.3f}, {normal_y:.3f}, {normal_z:.3f}]")
            
            # In sagittal imaging, the image normal is primarily along the patient's left-right axis
            # So the x-component of the normal should be large
            if abs(normal_x) > 0.8:
                if debug:
                    print(f"    PASS: Image plane orientation indicates sagittal view")
                return True
            
            # Another approach: check for sagittal using the fact that 
            # row vector should be along H->F and column vector along P->A
            if (abs(row_z) > 0.8 and abs(col_y) > 0.8) or (abs(row_y) > 0.8 and abs(col_z) > 0.8):
                if debug:
                    print(f"    PASS: Row/column vectors suggest sagittal orientation")
                return True
                
        ## If this is a spine series without clear orientation, let's process it anyway
        if is_spine and not hasattr(dicom, 'ImageOrientationPatient'):
            if debug:
                print(f"    PASS: Spine series without orientation info - assuming sagittal")
            return True
            
        # Look at slice spacing as a last resort
        if len(dicom_files) > 10:
            # Try a few more files to see if we can determine the orientation
            sorted_files = sorted(dicom_files)
            positions = []
            
            for f in sorted_files[:min(5, len(sorted_files))]:
                try:
                    dcm = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True)
                    if hasattr(dcm, 'ImagePositionPatient'):
                        positions.append(dcm.ImagePositionPatient)
                except Exception:
                    continue
                    
            if len(positions) >= 2:
                # Check which axis has the most variation between slices
                max_diffs = [0, 0, 0]
                for i in range(len(positions)-1):
                    for axis in range(3):
                        diff = abs(positions[i+1][axis] - positions[i][axis])
                        max_diffs[axis] = max(max_diffs[axis], diff)
                        
                # In sagittal series, the x coordinate should change the most between slices
                if max_diffs[0] > max_diffs[1] and max_diffs[0] > max_diffs[2]:
                    if debug:
                        print(f"    PASS: Slice positions suggest sagittal orientation")
                    return True
        
        if debug:
            print(f"    FAIL: No indicators of sagittal orientation found")
            
    except Exception as e:
        if debug:
            print(f"    Error reading DICOM file {dicom_path}: {e}")
    
    return False 