#!/usr/bin/env python3
"""
Design C - PyTorch VRN Architecture
Volumetric CNN for 3D face reconstruction (200×192×192 occupancy volume output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3DBlock(nn.Module):
    """3D Convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VRNPyTorch(nn.Module):
    """
    PyTorch implementation of VRN-style volumetric face reconstruction
    
    Architecture:
    - 2D encoder: Extract image features
    - 2D→3D projection: Lift features to volume
    - 3D decoder: Refine volumetric representation
    - Output: 200×192×192 occupancy volume
    
    Input: (B, 3, H, W) RGB image
    Output: (B, 1, 200, 192, 192) occupancy logits
    """
    
    def __init__(self, input_size=(224, 224), volume_shape=(200, 192, 192)):
        super().__init__()
        
        self.input_size = input_size
        self.volume_shape = volume_shape
        
        # === 2D Encoder (image → features) ===
        # Progressive downsampling: 224→112→56→28→14→7
        self.encoder = nn.ModuleList([
            # Block 1: 224×224
            ConvBlock(3, 32, stride=2),      # → 112×112, 32 channels
            ConvBlock(32, 32),
            
            # Block 2: 112×112
            ConvBlock(32, 64, stride=2),     # → 56×56, 64 channels
            ConvBlock(64, 64),
            
            # Block 3: 56×56
            ConvBlock(64, 128, stride=2),    # → 28×28, 128 channels
            ConvBlock(128, 128),
            
            # Block 4: 28×28
            ConvBlock(128, 256, stride=2),   # → 14×14, 256 channels
            ConvBlock(256, 256),
            
            # Block 5: 14×14
            ConvBlock(256, 512, stride=2),   # → 7×7, 512 channels
            ConvBlock(512, 512),
        ])
        
        # === 2D→3D Projection ===
        # Lift 7×7×512 feature map to initial volume
        # Target: start with small volume (e.g., 25×24×24) then upsample
        # Reduced to 32 channels to save memory
        self.projection = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 25 * 24 * 24 * 32),  # Initial volume: 25×24×24, 32 channels
            nn.ReLU(inplace=True)
        )
        
        # === 3D Decoder (volume refinement + upsampling) ===
        # Upsample 25×24×24 → 200×192×192
        # Factors: X: 8x (25→200), Y: 8x (24→192), Z: 8x (24→192)
        # Memory-efficient: fewer channels throughout
        
        self.decoder = nn.ModuleList([
            # Initial volume: 25×24×24×32
            Conv3DBlock(32, 32),
            
            # Upsample stage 1: → 50×48×48
            nn.ConvTranspose3d(32, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            Conv3DBlock(24, 24),
            
            # Upsample stage 2: → 100×96×96
            nn.ConvTranspose3d(24, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            Conv3DBlock(16, 16),
            
            # Upsample stage 3: → 200×192×192
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            Conv3DBlock(8, 8),
        ])
        
        # Final output layer
        self.output = nn.Conv3d(8, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image
        
        Returns:
            logits: (B, 1, 200, 192, 192) occupancy logits
        """
        batch_size = x.size(0)
        
        # === Encoder: 2D features ===
        for layer in self.encoder:
            x = layer(x)
        
        # x: (B, 512, 7, 7)
        
        # === Projection: 2D → 3D ===
        x = x.view(batch_size, -1)  # Flatten: (B, 512*7*7)
        x = self.projection(x)  # (B, 25*24*24*64)
        
        # Reshape to 3D volume
        x = x.view(batch_size, 32, 25, 24, 24)  # (B, C, X, Y, Z)
        
        # === Decoder: 3D volume refinement + upsampling ===
        for layer in self.decoder:
            x = layer(x)
        
        # x: (B, 8, 200, 192, 192)
        
        # === Output: occupancy logits ===
        logits = self.output(x)  # (B, 1, 200, 192, 192)
        
        return logits
    
    def predict(self, x, threshold=0.5):
        """
        Predict binary occupancy volume
        
        Args:
            x: (B, 3, H, W) RGB image
            threshold: Occupancy threshold (default: 0.5, matching Design B)
        
        Returns:
            volume: (B, 200, 192, 192) boolean occupancy
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            volume = (probs > threshold).squeeze(1)  # Remove channel dim
        
        return volume


def test_model():
    """Test model architecture"""
    print("Testing VRNPyTorch architecture...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = VRNPyTorch(input_size=(224, 224), volume_shape=(200, 192, 192))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        logits = model(x)
        print(f"Output logits shape: {logits.shape}")
        
        volume = model.predict(x, threshold=0.5)
        print(f"Output volume shape: {volume.shape}")
        print(f"Output volume dtype: {volume.dtype}")
        
        occupancy = volume.float().mean() * 100
        print(f"Average occupancy: {occupancy:.2f}%")
    
    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()
