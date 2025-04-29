#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Slice classification model for spinal canal level detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any


class SliceClassificationModel(nn.Module):
    """
    Model for classifying which slice best represents a specific spinal level.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.2,
        num_levels: int = 5
    ):
        """
        Initialize the model.
        
        Args:
            backbone: Backbone architecture ("resnet18", "resnet34", "resnet50")
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            dropout_rate: Dropout rate
            num_levels: Number of spinal levels to detect
        """
        super().__init__()
        
        self.backbone = backbone
        self.in_channels = in_channels
        self.num_levels = num_levels
        
        # Define level names for reference
        self.level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        
        # Create backbone network
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
        
        # Create classification heads for each level
        self.classification_heads = nn.ModuleList()
        
        for _ in range(self.num_levels):
            # Stronger classifier architecture with more layers and attention mechanism
            classifier = nn.Sequential(
                # Spatial attention mechanism
                nn.Conv2d(self.feature_size, self.feature_size, kernel_size=1),
                nn.Sigmoid(),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),

                # MLP with more capacity
                nn.Linear(self.feature_size, 512),
                nn.LayerNorm(512),  # LayerNorm works with batch size 1
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(512, 256),
                nn.LayerNorm(256),  # LayerNorm works with batch size 1
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 1)  # Binary classification: is this the optimal slice?
            )
            self.classification_heads.append(classifier)
    
    def forward(self, x, level_idx=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            level_idx: Index of level to predict (None means predict all levels)
            
        Returns:
            If level_idx is None: Dictionary of classification scores for each level
            If level_idx is specified: Classification score for the specified level
        """
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
        """
        Forward pass that returns tensor with all level predictions stacked.
        
        Args:
            x: Input tensor [B, C, H, W]
            apply_sigmoid: Whether to apply sigmoid to outputs
            
        Returns:
            Tensor with classification scores [B, num_levels]
        """
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
