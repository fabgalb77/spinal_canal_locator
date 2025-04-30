"""
Model definition for spine classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Define constants
LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

class SimpleSpineModel(nn.Module):
    """
    Simplified model for spine slice classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.5
    ):
        """Initialize the model with a minimal backbone."""
        super().__init__()
        
        self.in_channels = in_channels
        
        # Simple feature extractor
        self.encoder = nn.Sequential(
            # First layer
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
        
        # Create classification heads for each level
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)
            ) for _ in range(len(LEVEL_NAMES))
        ])
    
    def forward(self, x, level_idx=None, return_logits=False):
        """
        Forward pass with option to get specific level or all levels.
        Now returns both logits and probabilities if return_logits=True.
        """
        # Extract features using shared backbone
        features = self.encoder(x)
        
        if level_idx is not None:
            if isinstance(level_idx, torch.Tensor) and level_idx.dim() > 0:
                # Handle batch of level indices
                batch_size = level_idx.size(0)
                logits = []
                
                for i in range(batch_size):
                    # Get single level index for this sample
                    idx = level_idx[i].item()
                    # Get feature for this sample
                    feat = features[i:i+1]
                    # Get classification output
                    out = self.classification_heads[idx](feat)
                    logits.append(out)
                
                logits = torch.cat(logits, dim=0)
            else:
                # Single level index
                logits = self.classification_heads[level_idx](features)
        else:
            # Generate classification outputs for all levels
            logits = torch.cat([head(features) for head in self.classification_heads], dim=1)
        
        if return_logits:
            return logits, torch.sigmoid(logits)
        else:
            return logits


class ResNetSpineModel(nn.Module):
    """
    More sophisticated model for spine slice classification using ResNet18 backbone.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        pretrained: bool = True
    ):
        """Initialize the model with ResNet18 backbone."""
        super().__init__()
        
        self.in_channels = in_channels
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept single-channel inputs if needed
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
            # If using pretrained weights, we need to adapt the first layer
            if pretrained:
                # Initialize the new first layer with mean across the RGB channels
                with torch.no_grad():
                    # Get the original weight
                    original_weight = self.resnet.conv1.weight.data.clone()
                    
                    # Average across the RGB channels and use as new single-channel weight
                    self.resnet.conv1.weight.data = original_weight.sum(dim=1, keepdim=True)
        
        # Get number of features in the final layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer with identity
        self.resnet.fc = nn.Identity()
        
        # Create classification heads for each level
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 1)
            ) for _ in range(len(LEVEL_NAMES))
        ])
    
    def forward(self, x, level_idx=None, return_logits=False):
        """
        Forward pass with option to get specific level or all levels.
        Returns both logits and probabilities if return_logits=True.
        """
        # Extract features using ResNet backbone
        features = self.resnet(x)
        
        if level_idx is not None:
            if isinstance(level_idx, torch.Tensor) and level_idx.dim() > 0:
                # Handle batch of level indices
                batch_size = level_idx.size(0)
                logits = []
                
                for i in range(batch_size):
                    # Get single level index for this sample
                    idx = level_idx[i].item()
                    # Get feature for this sample
                    feat = features[i:i+1]
                    # Get classification output
                    out = self.classification_heads[idx](feat)
                    logits.append(out)
                
                logits = torch.cat(logits, dim=0)
            else:
                # Single level index
                logits = self.classification_heads[level_idx](features)
        else:
            # Generate classification outputs for all levels
            logits = torch.cat([head(features) for head in self.classification_heads], dim=1)
        
        if return_logits:
            return logits, torch.sigmoid(logits)
        else:
            return logits