#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slice classification model for spinal canal level detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any

class SimpleSpineModel(nn.Module):
    """
    Extremely simplified model for spine slice classification to prevent overfitting.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        num_levels: int = 5
    ):
        """Initialize the model."""
        super().__init__()
        
        self.in_channels = in_channels
        self.num_levels = num_levels
        
        # Define level names for reference
        self.level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        
        # Very simple feature extractor with strong regularization
        self.encoder = nn.Sequential(
            # First layer (dramatically simplified)
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second layer
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Final layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Create simple classification heads for each level
        self.classification_heads = nn.ModuleList()
        
        for _ in range(self.num_levels):
            classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)  # Single output per level
            )
            self.classification_heads.append(classifier)
    
    def forward(self, x, level_idx=None):
        """Forward pass."""
        # Extract features using shared backbone
        features = self.encoder(x)
        
        if level_idx is not None:
            # Get classification output for specific level
            classification = self.classification_heads[level_idx](features)
            return classification
        else:
            # Generate classification outputs for all levels
            classifications = {}
            
            for i in range(self.num_levels):
                # Get classification output
                classification = self.classification_heads[i](features)
                classifications[f"level_{i}"] = classification
            
            return classifications
    
    def forward_all_levels(self, x, apply_sigmoid=False):
        """Forward pass that returns tensor with all level predictions stacked."""
        # Extract features using shared backbone
        features = self.encoder(x)
        
        # Generate classification scores for all levels
        classifications = []
        
        for i in range(self.num_levels):
            # Get classification output
            classification = self.classification_heads[i](features)
            classifications.append(classification)
        
        # Stack along feature dimension for classifications
        stacked_classifications = torch.cat(classifications, dim=1)
        
        # Apply sigmoid if requested
        if apply_sigmoid:
            stacked_classifications = torch.sigmoid(stacked_classifications)
        
        return stacked_classifications
    
    def get_level_name(self, level_idx):
        """Get name of a specific level."""
        return self.level_names[level_idx]


class SliceClassificationModel(nn.Module):
    """
    Simplified model for classifying which slice best represents a specific spinal level.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
        num_levels: int = 5
    ):
        """Initialize the model."""
        super().__init__()
        
        self.backbone = backbone
        self.in_channels = in_channels
        self.num_levels = num_levels
        
        # Define level names for reference
        self.level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        
        # Create backbone network with modified first layer if needed
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_size = 512
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_size = 512
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first layer if input channels != 3
        if in_channels != 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(base_model.children())[1:-2]  # Skip first conv and final FC layer
            )
        else:
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])  # Remove final FC layer
        
        # Create simplified classification heads for each level
        self.classification_heads = nn.ModuleList()
        
        for _ in range(self.num_levels):
            # Much simpler classifier - just pooling and a single fully connected layer
            classifier = nn.Sequential(
                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # Simple dropout + single linear layer
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_size, 1)
            )
            self.classification_heads.append(classifier)
    
    def forward(self, x, level_idx=None):
        """Forward pass."""
        # Extract features using shared backbone
        features = self.encoder(x)
        
        if level_idx is not None:
            # Get classification output for specific level
            classification = self.classification_heads[level_idx](features)
            return classification
        else:
            # Generate classification outputs for all levels
            classifications = {}
            
            for i in range(self.num_levels):
                # Get classification output
                classification = self.classification_heads[i](features)
                classifications[f"level_{i}"] = classification
            
            return classifications
    
    def forward_all_levels(self, x, apply_sigmoid=False):
        """Forward pass that returns tensor with all level predictions stacked."""
        # Extract features using shared backbone
        features = self.encoder(x)
        
        # Generate classification scores for all levels
        classifications = []
        
        for i in range(self.num_levels):
            # Get classification output
            classification = self.classification_heads[i](features)
            classifications.append(classification)
        
        # Stack along feature dimension for classifications
        stacked_classifications = torch.cat(classifications, dim=1)
        
        # Apply sigmoid if requested
        if apply_sigmoid:
            stacked_classifications = torch.sigmoid(stacked_classifications)
        
        return stacked_classifications
    
    def get_level_name(self, level_idx):
        """Get name of a specific level."""
        return self.level_names[level_idx]
