import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    """Bottleneck residual block used throughout hourglass network"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels//2)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += residual
        
        return out


class HourglassModule(nn.Module):
    """Single hourglass module with recursive structure"""
    def __init__(self, in_channels, out_channels, depth=4, debug=False):
        super(HourglassModule, self).__init__()
        self.depth = depth
        self.debug = debug
        
        # Upper branch (skip connection)
        self.skip = ResidualBlock(in_channels, out_channels)
        
        # Lower branch (encoder-decoder)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_residual = ResidualBlock(in_channels, out_channels)
        
        # Recursive hourglass or bottom block
        if self.depth > 1:
            self.hourglass = HourglassModule(out_channels, out_channels, depth-1, debug=debug)
        else:
            self.hourglass = ResidualBlock(out_channels, out_channels)
        
        # Up branch
        self.up_residual = ResidualBlock(out_channels, out_channels)
        # CHANGED: Use bilinear upsampling instead of nearest for better spatial precision
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        if self.debug:
            print(f"HG (depth {self.depth}) input shape: {x.shape}")
        
        # Upper branch
        up1 = self.skip(x)
        if self.debug:
            print(f"HG (depth {self.depth}) skip connection shape: {up1.shape}")
        
        # Lower branch
        low1 = self.max_pool(x)
        low1 = self.down_residual(low1)
        if self.debug:
            print(f"HG (depth {self.depth}) downsampled shape: {low1.shape}")
        
        # Recursive or bottom
        low2 = self.hourglass(low1)
        if self.debug:
            print(f"HG (depth {self.depth}) hourglass output shape: {low2.shape}")
        
        # Up branch
        low3 = self.up_residual(low2)
        if self.debug:
            print(f"HG (depth {self.depth}) up residual shape: {low3.shape}")
            
        up2 = self.up_sample(low3)
        if self.debug:
            print(f"HG (depth {self.depth}) upsampled shape: {up2.shape}")
            print(f"HG (depth {self.depth}) skip shape to add: {up1.shape}")
        
        # IMPROVED: Always ensure dimensions match precisely
        if up1.size() != up2.size():
            if self.debug:
                print(f"HG (depth {self.depth}) resizing upsampled from {up2.shape} to {up1.shape[2:]}")
            up2 = F.interpolate(up2, size=(up1.size(2), up1.size(3)), mode='bilinear', align_corners=True)
        
        # Visualize feature maps in debug mode for first batch item
        if self.debug and self.depth == 4:  # Only for the top-level module
            with torch.no_grad():
                # Visualize skip connection (up1)
                self._visualize_feature_map(up1[0], "skip_connection")
                # Visualize upsampled features (up2)
                self._visualize_feature_map(up2[0], "upsampled_features")
                # Visualize their sum
                self._visualize_feature_map((up1 + up2)[0], "combined_features")
        
        return up1 + up2
    
    def _visualize_feature_map(self, feature_map, name):
        # Select first few channels to visualize
        num_channels = min(4, feature_map.size(0))
        plt.figure(figsize=(12, 3))
        
        for i in range(num_channels):
            plt.subplot(1, num_channels, i+1)
            plt.imshow(feature_map[i].cpu().numpy(), cmap='viridis')
            plt.title(f"Channel {i}")
            plt.axis('off')
        
        plt.suptitle(f"Feature Map: {name}")
        plt.tight_layout()
        plt.savefig(f"debug_feature_{name}.png")
        plt.close()


class StackedHourglassNetwork(nn.Module):
    """Complete stacked hourglass network for spinal canal localization"""
    def __init__(self, input_channels=3, num_stacks=2, num_blocks=4, num_classes=5, 
                 heatmap_size=128, debug=False):
        super(StackedHourglassNetwork, self).__init__()
        
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size
        self.debug = debug
        self.level_names = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
        
        # Initial processing
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256)
        )
        
        # Stacks of hourglasses with debug flag
        self.hourglass_modules = nn.ModuleList([
            HourglassModule(256, 256, depth=num_blocks, debug=debug) 
            for _ in range(num_stacks)
        ])
        
        # Features
        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(256, 256),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stacks)
        ])
        
        # IMPROVED: Output heatmaps with batch normalization
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.BatchNorm2d(num_classes)
            ) for _ in range(num_stacks)
        ])
        
        # Intermediate supervision
        if num_stacks > 1:
            self.merge_features = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1)
                for _ in range(num_stacks-1)
            ])
            
            self.merge_preds = nn.ModuleList([
                nn.Conv2d(num_classes, 256, kernel_size=1)
                for _ in range(num_stacks-1)
            ])
            
    def forward(self, x, level_idx=None):
        """
        Forward pass for training and inference
        Args:
            x: Input image tensor
            level_idx: Level index to return (if None, returns all levels)
        """
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Initial feature extraction
        x = self.initial(x)
        if self.debug:
            print(f"After initial feature extraction: {x.shape}")
        
        # Stacked hourglass inference
        outputs = []
        
        for i in range(self.num_stacks):
            if self.debug:
                print(f"\nStack {i+1}:")
                
            # Process through hourglass
            hg = self.hourglass_modules[i](x)
            if self.debug:
                print(f"After hourglass module {i+1}: {hg.shape}")
            
            # Get features
            features = self.features[i](hg)
            if self.debug:
                print(f"After feature processing: {features.shape}")
            
            # Get predictions (heatmaps)
            preds = self.outputs[i](features)
            if self.debug:
                print(f"Raw heatmap predictions: {preds.shape}")
                
            outputs.append(preds)
            
            # If not last stack, prepare for next stack
            if i < self.num_stacks - 1:
                merged_features = self.merge_features[i](features)
                merged_preds = self.merge_preds[i](preds)
                
                if self.debug:
                    print(f"Merged features: {merged_features.shape}")
                    print(f"Merged predictions: {merged_preds.shape}")
                    
                x = x + merged_features + merged_preds
                if self.debug:
                    print(f"Input to next stack: {x.shape}")
        
        # Get final output from last stack
        final_output = outputs[-1]  # Shape: (B, num_classes, H', W')
        
        # IMPROVED: Check output size before resizing
        curr_size = (final_output.shape[2], final_output.shape[3])
        target_size = (self.heatmap_size, self.heatmap_size)
        
        if self.debug:
            print(f"Current output size: {curr_size}, Target size: {target_size}")
        
        # Resize to desired heatmap size if needed
        if curr_size != target_size:
            if self.debug:
                print(f"Resizing output from {curr_size} to {target_size}")
                
            final_output = F.interpolate(
                final_output, 
                size=target_size, 
                mode='bilinear', 
                align_corners=True
            )
        
        # IMPROVED: Apply sigmoid to get proper heatmap values
        # This is critical for correct heatmap interpretation
        final_output = torch.sigmoid(final_output)
        
        if self.debug:
            self._visualize_full_heatmaps(final_output[0], "final_heatmaps")
        
        # Return specific level if requested
        if level_idx is not None:
            if isinstance(level_idx, int):
                return final_output[:, level_idx:level_idx+1]
            elif isinstance(level_idx, torch.Tensor):
                batch_size = final_output.size(0)
                selected_heatmaps = torch.zeros(
                    (batch_size, 1, self.heatmap_size, self.heatmap_size), 
                    device=x.device
                )
                for b in range(batch_size):
                    idx = level_idx[b].item() if torch.is_tensor(level_idx[b]) else level_idx[b]
                    selected_heatmaps[b, 0] = final_output[b, idx]
                return selected_heatmaps
        
        return final_output
    
    def _visualize_full_heatmaps(self, heatmaps, name):
        """Visualize all heatmaps for debugging"""
        plt.figure(figsize=(15, 3))
        for i in range(self.num_classes):
            plt.subplot(1, self.num_classes, i+1)
            plt.imshow(heatmaps[i].cpu().detach().numpy(), cmap='hot')
            plt.title(f"Level {self.level_names[i]}")
            plt.colorbar()
            plt.axis('off')
        
        plt.suptitle(f"{name}")
        plt.tight_layout()
        plt.savefig(f"debug_{name}.png")
        plt.close()
    
    def forward_all_levels(self, x, apply_sigmoid=True):
        """
        Forward pass that returns heatmaps for all levels (for visualization)
        """
        # Get all stacks output, use final stack
        outputs = []
        
        # Initial feature extraction
        x = self.initial(x)
        
        # Stacked hourglass inference
        for i in range(self.num_stacks):
            hg = self.hourglass_modules[i](x)
            features = self.features[i](hg)
            preds = self.outputs[i](features)
            outputs.append(preds)
            
            if i < self.num_stacks - 1:
                x = x + self.merge_features[i](features) + self.merge_preds[i](preds)
        
        final_output = outputs[-1]
        
        # Resize if needed
        if final_output.shape[2] != self.heatmap_size or final_output.shape[3] != self.heatmap_size:
            final_output = F.interpolate(
                final_output, 
                size=(self.heatmap_size, self.heatmap_size), 
                mode='bilinear', 
                align_corners=True
            )
        
        # IMPROVED: Always apply sigmoid for proper heatmap visualization unless explicitly disabled
        if apply_sigmoid:
            final_output = torch.sigmoid(final_output)
            
        return final_output

    def get_level_name(self, level_idx):
        """Get name of a specific level."""
        return self.level_names[level_idx]

    # ADDED: Method to visualize intermediate features
    def visualize_features(self, x, level_idx=0):
        """
        Visualize intermediate features through the network for debugging
        """
        print("Visualizing features through the network...")
        
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Initial features
            initial_features = self.initial(x)
            self._visualize_feature_map(initial_features[0], "initial_features")
            
            # First hourglass
            hg_output = self.hourglass_modules[0](initial_features)
            self._visualize_feature_map(hg_output[0], "hourglass_output")
            
            # After feature processing
            features = self.features[0](hg_output)
            self._visualize_feature_map(features[0], "processed_features")
            
            # Final output for specific level
            heatmaps = self.outputs[0](features)
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmaps[0, level_idx].cpu().numpy(), cmap='hot')
            plt.title(f"Raw Heatmap: {self.level_names[level_idx]}")
            plt.colorbar()
            plt.savefig(f"debug_raw_heatmap_{level_idx}.png")
            plt.close()
            
            # Final output after sigmoid
            heatmaps_sigmoid = torch.sigmoid(heatmaps)
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmaps_sigmoid[0, level_idx].cpu().numpy(), cmap='hot')
            plt.title(f"Sigmoid Heatmap: {self.level_names[level_idx]}")
            plt.colorbar()
            plt.savefig(f"debug_sigmoid_heatmap_{level_idx}.png")
            plt.close()
            
        print("Feature visualization complete. Check debug_* images.")
    
    def _visualize_feature_map(self, feature_map, name):
        # Select first few channels to visualize
        num_channels = min(4, feature_map.size(0))
        plt.figure(figsize=(12, 3))
        
        for i in range(num_channels):
            plt.subplot(1, num_channels, i+1)
            plt.imshow(feature_map[i].cpu().numpy(), cmap='viridis')
            plt.title(f"Channel {i}")
            plt.axis('off')
        
        plt.suptitle(f"Feature Map: {name}")
        plt.tight_layout()
        plt.savefig(f"debug_feature_{name}.png")
        plt.close()


def create_spinal_canal_hourglass(config):
    """
    Factory function to create a hourglass model from config
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        StackedHourglassNetwork model
    """
    input_channels = config['model'].get('input_channels', 3)
    num_stacks = config['model'].get('num_stacks', 2)
    num_blocks = config['model'].get('num_blocks', 4)
    heatmap_size = config['model'].get('heatmap_size', 128)
    debug = config.get('debug', False)
    
    return StackedHourglassNetwork(
        input_channels=input_channels,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        heatmap_size=heatmap_size,
        debug=debug
    )