#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing for spinal canal localization.
"""

import os
import pandas as pd
import numpy as np
import logging
import pydicom
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Set
import time


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def find_sagittal_series(series_csv: str) -> Set[str]:
    """
    Identify sagittal series based on series descriptions.
    
    Args:
        series_csv: Path to CSV file with series descriptions
        
    Returns:
        Set of series IDs that are likely sagittal
    """
    series_df = pd.read_csv(series_csv)
    
    # Filter for sagittal series
    sagittal_series = series_df[
        series_df['series_description'].str.contains('Sagittal', case=False)
    ]
    
    return set(sagittal_series['series_id'].astype(str))


def find_canal_annotations(coordinates_csv: str) -> Set[Tuple[str, str]]:
    """
    Identify series with spinal canal annotations.
    
    Args:
        coordinates_csv: Path to CSV file with coordinate annotations
        
    Returns:
        Set of (study_id, series_id) tuples with canal annotations
    """
    coordinates_df = pd.read_csv(coordinates_csv)
    
    # Filter for spinal canal annotations
    canal_df = coordinates_df[
        coordinates_df['condition'].str.contains('Spinal Canal Stenosis', case=False)
    ]
    
    # Group by study_id and series_id
    study_series_pairs = set(
        (str(row['study_id']), str(row['series_id'])) 
        for _, row in canal_df.iterrows()
    )
    
    return study_series_pairs


def check_file_exists(data_dir: str, study_id: str, series_id: str, instance_number: int) -> bool:
    """
    Check if a DICOM file exists.
    
    Args:
        data_dir: Root directory for DICOM files
        study_id: Study ID
        series_id: Series ID
        instance_number: Instance number
        
    Returns:
        True if file exists, False otherwise
    """
    file_path = os.path.join(data_dir, study_id, series_id, f"{instance_number}.dcm")
    return os.path.exists(file_path)


def get_optimal_slices_for_canal(
    coordinates_csv: str, 
    data_dir: str, 
    sagittal_series: Set[str],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[int]]]]]:
    """
    Find optimal slices for each spinal canal level in each series.
    
    Args:
        coordinates_csv: Path to CSV file with coordinate annotations
        data_dir: Root directory for DICOM files
        sagittal_series: Set of sagittal series IDs
        logger: Logger instance
        
    Returns:
        Nested dictionary mapping study_id -> series_id -> level -> {'instances': [inst_nums], 'coordinates': [(x, y)]}
    """
    if logger is None:
        logger = setup_logger('optimal_slices')
    
    coordinates_df = pd.read_csv(coordinates_csv)
    
    # Filter for spinal canal annotations
    canal_df = coordinates_df[
        (coordinates_df['condition'].str.contains('Spinal Canal Stenosis', case=False)) &
        (coordinates_df['series_id'].astype(str).isin(sagittal_series))
    ]
    
    # Dictionary to store optimal slices
    optimal_slices = {}
    
    # Group by study, series, and level
    for (study_id, series_id, level), group in canal_df.groupby(['study_id', 'series_id', 'level']):
        study_id = str(study_id)
        series_id = str(series_id)
        
        # Skip if study/series doesn't exist
        if not os.path.exists(os.path.join(data_dir, study_id, series_id)):
            continue
        
        # Make sure we have dictionaries for each level
        if study_id not in optimal_slices:
            optimal_slices[study_id] = {}
        if series_id not in optimal_slices[study_id]:
            optimal_slices[study_id][series_id] = {}
        
        # Calculate "quality" of each instance for this level
        # For now, we'll just use all available instances
        instances = []
        coordinates = []
        
        for _, row in group.iterrows():
            instance_number = int(row['instance_number'])
            
            # Check if file exists
            if check_file_exists(data_dir, study_id, series_id, instance_number):
                instances.append(instance_number)
                coordinates.append((float(row['x']), float(row['y'])))
        
        # Store the instances for this level
        if instances:
            optimal_slices[study_id][series_id][level] = {
                'instances': instances,
                'coordinates': coordinates
            }
    
    # Log some statistics
    total_studies = len(optimal_slices)
    total_series = sum(len(series_dict) for series_dict in optimal_slices.values())
    total_levels = sum(
        sum(len(level_dict) for level_dict in series_dict.values())
        for series_dict in optimal_slices.values()
    )
    
    logger.info(f"Found {total_studies} studies with canal annotations")
    logger.info(f"Found {total_series} series with canal annotations")
    logger.info(f"Found {total_levels} level annotations across all series")
    
    return optimal_slices


def create_processed_dataset(
    data_dir: str,
    series_csv: str,
    coordinates_csv: str,
    output_dir: str,
    force_reprocess: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Create a processed dataset for spinal canal localization.
    
    Args:
        data_dir: Root directory for DICOM files
        series_csv: Path to CSV file with series descriptions
        coordinates_csv: Path to CSV file with coordinate annotations
        output_dir: Directory to save processed files
        force_reprocess: Whether to force reprocessing even if processed files exist
        logger: Logger instance
        
    Returns:
        Dictionary with paths to processed files
    """
    if logger is None:
        logger = setup_logger('preprocessing')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths for processed files
    canal_slices_path = os.path.join(output_dir, "canal_optimal_slices.json")
    metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
    
    # Check if processed files already exist
    if (os.path.exists(canal_slices_path) and
        os.path.exists(metadata_path) and
        not force_reprocess):
        
        # Validate metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if (metadata.get('data_dir') == data_dir and
                metadata.get('series_csv') == series_csv and
                metadata.get('coordinates_csv') == coordinates_csv):
                
                logger.info("Using existing processed files")
                return {
                    "canal_slices": canal_slices_path,
                    "metadata": metadata_path
                }
        except Exception as e:
            logger.warning(f"Error validating metadata: {e}")
    
    # If we get here, we need to preprocess
    logger.info("Starting preprocessing...")
    start_time = time.time()
    
    try:
        # Find sagittal series
        logger.info("Finding sagittal series...")
        sagittal_series = find_sagittal_series(series_csv)
        logger.info(f"Found {len(sagittal_series)} sagittal series")
        
        # Find optimal slices for canal localization
        logger.info("Finding optimal slices for canal localization...")
        canal_slices = get_optimal_slices_for_canal(
            coordinates_csv=coordinates_csv,
            data_dir=data_dir,
            sagittal_series=sagittal_series,
            logger=logger
        )
        
        # Save processed files
        with open(canal_slices_path, 'w') as f:
            json.dump(canal_slices, f, indent=2)
        
        # Save metadata
        metadata = {
            "data_dir": data_dir,
            "series_csv": series_csv,
            "coordinates_csv": coordinates_csv,
            "timestamp": time.time(),
            "sagittal_series_count": len(sagittal_series),
            "canal_studies_count": len(canal_slices),
            "preprocessing_time": time.time() - start_time
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        return {
            "canal_slices": canal_slices_path,
            "metadata": metadata_path
        }
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Example usage
    logger = setup_logger('preprocessing')
    
    data_dir = "/path/to/train_images"  # Replace with your path
    series_csv = "/path/to/train_series_descriptions.csv"  # Replace with your path
    coordinates_csv = "/path/to/train_label_coordinates.csv"  # Replace with your path
    output_dir = "./processed_data"
    
    result = create_processed_dataset(
        data_dir=data_dir,
        series_csv=series_csv,
        coordinates_csv=coordinates_csv,
        output_dir=output_dir,
        force_reprocess=False,
        logger=logger
    )
    
    if result:
        logger.info(f"Processed files saved to {output_dir}")
    else:
        logger.error("Preprocessing failed")
