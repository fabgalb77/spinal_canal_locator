#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dedicated localization model architecture for spinal canal localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any


class SpinalCanalLocalizationModel(nn.Module):
    """
    Model focused solely on localizing the spinal canal at different levels.
    This is a refined version of the localization part from the dual-headed model.
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
        
        # Create separate localization decoder heads for each spinal level
        self.localization_decoders = nn.ModuleList()
        
        for _ in range(self.num_levels):
            decoder = nn.Sequential(
                # First upsampling block
                nn.ConvTranspose2d(self.feature_size, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                
                # Second upsampling block
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                
                # Third upsampling block
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Fourth upsampling block
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Fifth upsampling block
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                
                # Final layer to produce single-channel heatmap
                nn.Conv2d(16, 1, kernel_size=1)
            )
            self.localization_decoders.append(decoder)
    
    def forward(self, x, level_idx=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            level_idx: Index of level to predict (None means predict all levels)
            
        Returns:
            If level_idx is None: Dictionary of heatmaps for each level
            If level_idx is specified: Heatmap for the specified level
        """
        # Extract features using shared backbone
        features = self.encoder(x)
        
        if level_idx is not None:
            # Get localization output for specific level
            heatmap = self.localization_decoders[level_idx](features)
            return heatmap
        else:
            # Generate outputs for all levels
            results = {}
            
            for i in range(self.num_levels):
                # Get localization output
                heatmap = self.localization_decoders[i](features)
                results[f"level_{i}"] = heatmap
            
            return results
    
    def forward_all_levels(self, x, apply_sigmoid=False):
        """
        Forward pass that returns tensor with all level heatmaps stacked.
        
        Args:
            x: Input tensor [B, C, H, W]
            apply_sigmoid: Whether to apply sigmoid to outputs
            
        Returns:
            Tensor with heatmaps for all levels [B, num_levels, H, W]
        """
        # Extract features using shared backbone
        features = self.encoder(x)
        
        # Generate heatmaps for all levels
        heatmaps = []
        
        for i in range(self.num_levels):
            # Get localization output
            heatmap = self.localization_decoders[i](features)
            heatmaps.append(heatmap)
        
        # Stack along channel dimension for heatmaps
        stacked_heatmaps = torch.cat(heatmaps, dim=1)
        
        # Apply sigmoid if requested
        if apply_sigmoid:
            stacked_heatmaps = torch.sigmoid(stacked_heatmaps)
        
        return stacked_heatmaps
    
    def get_level_name(self, level_idx):
        """Get name of a specific level."""
        return self.level_names[level_idx]
