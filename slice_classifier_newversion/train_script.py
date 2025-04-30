#!/usr/bin/env python3
"""
Simplified Spine Slice Classifier
---------------------------------
Single-slice approach with minimal backbone to reduce position bias and overfitting.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from custom modules
from model import SimpleSpineModel, ResNetSpineModel, LEVEL_NAMES
from dataset import SliceClassificationDataset, create_data_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_slice_classifier.log')
    ]
)
logger = logging.getLogger('train_slice_classifier')

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#############################################################
# Training and Validation Functions
#############################################################

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """Train model for one epoch."""
    model.train()
    
    # Initialize metrics
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    level_losses = {level: 0.0 for level in LEVEL_NAMES}
    level_counts = {level: 0 for level in LEVEL_NAMES}
    
    # Training loop
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Get batch data
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        level_indices = batch["level_idx"].to(device)
        
        # Forward pass
        outputs = model(images, level_indices)
        outputs = outputs.squeeze(1)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item() * images.size(0)
        
        # Get predictions and update level metrics
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        
        # Update level-specific metrics
        for i, level_idx in enumerate(level_indices):
            level = LEVEL_NAMES[level_idx]
            level_losses[level] += loss.item()
            level_counts[level] += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
    
    # Calculate final metrics
    epoch_loss /= len(dataloader.dataset)
    
    # Calculate AUC if possible
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except:
        epoch_auc = 0.0
    
    # Calculate AP if possible
    try:
        epoch_ap = average_precision_score(all_labels, all_preds)
    except:
        epoch_ap = 0.0
    
    # Calculate sensitivity and specificity
    threshold = 0.5
    preds_binary = (np.array(all_preds) > threshold).astype(int)
    labels_binary = np.array(all_labels).astype(int)
    
    # Calculate sensitivity (true positive rate)
    sensitivity = np.sum((preds_binary == 1) & (labels_binary == 1)) / max(1, np.sum(labels_binary == 1))
    
    # Calculate specificity (true negative rate)
    specificity = np.sum((preds_binary == 0) & (labels_binary == 0)) / max(1, np.sum(labels_binary == 0))
    
    # Calculate level-specific losses
    for level in LEVEL_NAMES:
        if level_counts[level] > 0:
            level_losses[level] /= level_counts[level]
    
    # Return metrics
    return {
        "loss": epoch_loss,
        "auc": epoch_auc,
        "ap": epoch_ap,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "level_losses": level_losses,
        "level_counts": level_counts
    }

def validate(model, dataloader, criterion, device):
    """Validate model on dataloader."""
    model.eval()
    all_losses = []
    all_predictions = []
    all_targets = []
    all_level_idxs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            level_idxs = batch["level_idx"].to(device)
            
            # Forward pass
            outputs = model(images, level_idxs)
            
            # Fix shape mismatch - ensure targets match output shape
            if outputs.shape != targets.shape:
                targets = targets.view(outputs.shape)  # Reshape to match outputs
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Store predictions and targets
            predictions = torch.sigmoid(outputs)
            
            all_losses.append(loss.item())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_level_idxs.append(level_idxs.cpu().numpy())
    
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_level_idxs = np.concatenate(all_level_idxs)
    
    # Flatten predictions and targets if needed
    all_predictions = all_predictions.flatten()
    all_targets = all_targets.flatten()
    
    # Calculate metrics using fixed 0.5 threshold
    binary_preds = (all_predictions >= 0.5).astype(int)
    accuracy = accuracy_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)
    
    # Calculate AUC and AP
    try:
        auc = roc_auc_score(all_targets, all_predictions)
        ap = average_precision_score(all_targets, all_predictions)
    except ValueError:
        # Handle case with only one class in targets
        auc = 0.5
        ap = np.mean(all_targets)
    
    # Calculate confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        sensitivity = 0
        specificity = 0
    
    # Calculate level-specific metrics
    level_metrics = {}
    for level_idx, level_name in enumerate(LEVEL_NAMES):
        level_mask = all_level_idxs == level_idx
        if np.sum(level_mask) > 0:
            level_predictions = all_predictions[level_mask]
            level_targets = all_targets[level_mask]
            level_binary_preds = binary_preds[level_mask]
            
            # Calculate metrics
            try:
                level_auc = roc_auc_score(level_targets, level_predictions)
            except ValueError:
                level_auc = 0.5
                
            level_accuracy = accuracy_score(level_targets, level_binary_preds)
            
            level_metrics[level_name] = {
                "auc": level_auc,
                "accuracy": level_accuracy,
                "threshold": 0.5,  # Fixed threshold
                "num_samples": np.sum(level_mask)
            }
    
    return {
        "loss": np.mean(all_losses),
        "auc": auc,
        "ap": ap,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": 0.5,  # Fixed threshold
        "level_metrics": level_metrics
    }

#############################################################
# Main Training Function
#############################################################

def train_model(
    data_dir: str,
    annotations_file: str,
    series_file: str, 
    output_dir: str,
    model_type: str = "simple",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: tuple = (256, 256),
    device_id: int = 0,
    num_workers: int = 4,
    save_best_only: bool = True,
    patience: int = 10,
    train_val_split: bool = True,
    train_ratio: float = 0.75,
    val_ratio: float = 0.25
):
    """Train a spine slice classifier with study-level train/val split."""
    # Set random seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms(target_size=image_size)
    
    # Check available studies in data_dir
    logger.info(f"Scanning available studies in {data_dir}")
    available_studies = []
    
    if os.path.exists(data_dir):
        # Get all directories in data_dir (these are study_ids)
        for study_dir in os.listdir(data_dir):
            study_path = os.path.join(data_dir, study_dir)
            if os.path.isdir(study_path):
                try:
                    study_id = int(study_dir)
                    available_studies.append(study_id)
                except ValueError:
                    # Skip directories that can't be converted to integers
                    pass
    
    logger.info(f"Found {len(available_studies)} available studies in data_dir")
    if available_studies:
        logger.info(f"Sample studies: {available_studies[:5]}")
    
    # Load and filter annotations
    annotations = pd.read_csv(annotations_file)
    condition = "Spinal Canal Stenosis"
    annotations = annotations[annotations["condition"] == condition]
    
    # Filter to only include available studies
    annotations = annotations[annotations["study_id"].isin(available_studies)]
    
    # Load series descriptions
    series = pd.read_csv(series_file)
    sagittal_series = series[series["series_description"].str.contains("Sagittal", case=False)]
    sagittal_ids = set(zip(sagittal_series["study_id"], sagittal_series["series_id"]))
    
    # Filter annotations to only include sagittal series
    annotations = annotations[
        annotations.apply(lambda row: (row["study_id"], row["series_id"]) in sagittal_ids, axis=1)
    ]
    
    # Get unique study IDs from filtered annotations
    study_ids = annotations["study_id"].unique()
    logger.info(f"Found {len(study_ids)} studies with sagittal canal stenosis annotations")
    
    if train_val_split:
        # Split at study level
        random.seed(42)
        random.shuffle(study_ids)
        
        # Calculate split indices
        train_end = int(len(study_ids) * train_ratio)
        
        # Split studies
        train_studies = study_ids[:train_end]
        val_studies = study_ids[train_end:]
        
        logger.info(f"Split studies: Train={len(train_studies)}, Val={len(val_studies)}")
        
        # Filter annotations for each split
        train_annotations = annotations[annotations["study_id"].isin(train_studies)]
        val_annotations = annotations[annotations["study_id"].isin(val_studies)]
        
        # Create datasets
        train_dataset = SliceClassificationDataset(
            data_dir=data_dir,
            annotations=train_annotations,
            series=sagittal_series,
            target_size=image_size,
            transform=train_transform,
            mode="train"
        )
        
        val_dataset = SliceClassificationDataset(
            data_dir=data_dir,
            annotations=val_annotations,
            series=sagittal_series,
            target_size=image_size,
            transform=val_transform,
            mode="val"
        )
    else:
        # Use all data (for final training)
        train_dataset = SliceClassificationDataset(
            data_dir=data_dir,
            annotations=annotations,
            series=sagittal_series,
            target_size=image_size,
            transform=train_transform,
            mode="train"
        )
        
        val_dataset = train_dataset  # Use same dataset for validation
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model based on model_type
    if model_type.lower() == 'resnet':
        logger.info("Using ResNetSpineModel with ResNet18 backbone")
        model = ResNetSpineModel(in_channels=1, dropout_rate=0.5, pretrained=True)
    else:
        logger.info("Using SimpleSpineModel with minimal backbone")
        model = SimpleSpineModel(in_channels=1, dropout_rate=0.5)
        
    model = model.to(device)
    
    # Create loss function with class balancing
    pos_weight = torch.tensor([2.]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Initialize best metrics
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Train model
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Log training metrics
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.6f}, "
            f"AUC: {train_metrics['auc']:.4f}, "
            f"AP: {train_metrics['ap']:.4f}, "
            f"Sensitivity: {train_metrics['sensitivity']:.4f}, "
            f"Specificity: {train_metrics['specificity']:.4f}"
        )
        
        # Log level-specific losses
        for level in LEVEL_NAMES:
            if level in train_metrics["level_losses"] and train_metrics["level_counts"][level] > 0:
                logger.info(
                    f"  Level {level}: "
                    f"Loss: {train_metrics['level_losses'][level]:.6f} "
                    f"({train_metrics['level_counts'][level]} samples)"
                )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log validation metrics
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {val_metrics['loss']:.6f}, "
            f"AUC: {val_metrics['auc']:.4f}, "
            f"AP: {val_metrics['ap']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, "
            f"Sensitivity: {val_metrics['sensitivity']:.4f}, "
            f"Specificity: {val_metrics['specificity']:.4f}"
        )
        
        # Log level-specific metrics
        for level, metrics in val_metrics["level_metrics"].items():
            logger.info(
                f"  Level {level}: "
                f"AUC: {metrics['auc']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f} "
                f"(Threshold: {metrics['threshold']:.4f}, "
                f"{metrics['num_samples']} samples)"
            )
        
        # Update learning rate scheduler
        scheduler.step(val_metrics["auc"])
        
        # Save model if it's the best so far
        if val_metrics["auc"] > best_val_auc:
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(output_dir, f"best_model_{model_type}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved at epoch {best_epoch}")
        else:
            patience_counter += 1
            
            # Save model if not save_best_only
            if not save_best_only:
                save_path = os.path.join(output_dir, f"model_{model_type}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(output_dir, f"best_model_{model_type}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved at epoch {best_epoch}")
        else:
            patience_counter += 1
            
            # Save model if not save_best_only
            if not save_best_only:
                save_path = os.path.join(output_dir, f"model_{model_type}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Training completed. Best model at epoch {best_epoch} with AUC {best_val_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a spine slice classifier")
    parser.add_argument("--data_dir", type=str, default="/mnt/c/users/Fabio Galbusera/Dropbox/RSNA/rsna-2024-lumbar-spine-degenerative-classification/train_images", help="Path to data directory")
    parser.add_argument("--annotations_file", type=str, default="/mnt/c/users/Fabio Galbusera/Dropbox/RSNA/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv", help="Path to annotations file")
    parser.add_argument("--series_file", type=str, default="/mnt/c/users/Fabio Galbusera/Dropbox/RSNA/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv", help="Path to series descriptions file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to output directory")
    parser.add_argument("--model_type", type=str, default="simple", choices=["simple", "resnet"], help="Model architecture to use: 'simple' or 'resnet'")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        annotations_file=args.annotations_file,
        series_file=args.series_file,
        output_dir=args.output_dir,
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=(args.image_size, args.image_size),
        device_id=args.device_id,
        num_workers=args.num_workers
    )