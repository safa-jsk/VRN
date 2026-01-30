#!/usr/bin/env python3
"""
Design C - Loss Functions
BCE, focal, and boundary-weighted losses for volumetric regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss3D(nn.Module):
    """
    Binary Cross-Entropy loss for volumetric occupancy prediction
    Wrapper around PyTorch's BCEWithLogitsLoss for 3D volumes
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, X, Y, Z) predicted occupancy logits
            target: (B, X, Y, Z) or (B, 1, X, Y, Z) ground truth bool/float
        
        Returns:
            loss: scalar
        """
        # Ensure target has channel dimension
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Convert target to float if boolean
        if target.dtype == torch.bool:
            target = target.float()
        
        return self.criterion(logits, target)


class FocalLoss3D(nn.Module):
    """
    Focal Loss for volumetric occupancy prediction
    Addresses class imbalance by down-weighting easy examples
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t: predicted probability for true class
    - α_t: class weight
    - γ: focusing parameter (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, X, Y, Z) predicted occupancy logits
            target: (B, X, Y, Z) or (B, 1, X, Y, Z) ground truth bool/float
        
        Returns:
            loss: scalar
        """
        # Ensure target has channel dimension
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Convert target to float if boolean
        if target.dtype == torch.bool:
            target = target.float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        # For positive examples: p_t = p, weight = (1-p)^γ
        # For negative examples: p_t = 1-p, weight = p^γ
        pt = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        )
        
        # Apply focal weighting
        focal_loss = focal_weight * bce_loss
        
        # Apply class weighting (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class BoundaryWeightedLoss3D(nn.Module):
    """
    Boundary-weighted BCE loss for volumetric occupancy
    Upweights voxels near surface boundaries to improve mesh quality
    """
    def __init__(self, boundary_weight=5.0, kernel_size=3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        
        # Create 3D Laplacian kernel for boundary detection
        # Simple approach: count neighbor differences
        self.register_buffer('laplacian_kernel', self._create_laplacian_kernel())
    
    def _create_laplacian_kernel(self):
        """Create 3D Laplacian kernel for boundary detection"""
        k = self.kernel_size
        kernel = torch.ones(1, 1, k, k, k)
        kernel[0, 0, k//2, k//2, k//2] = -((k**3) - 1)
        return kernel / (k**3)
    
    def _detect_boundaries(self, volume):
        """
        Detect surface boundaries using 3D convolution
        
        Args:
            volume: (B, 1, X, Y, Z) boolean/float volume
        
        Returns:
            boundary_mask: (B, 1, X, Y, Z) float weights (higher at boundaries)
        """
        # Convert to float
        vol_float = volume.float() if volume.dtype == torch.bool else volume
        
        # Apply Laplacian (edge detection)
        padding = self.kernel_size // 2
        edges = F.conv3d(vol_float, self.laplacian_kernel, padding=padding)
        
        # High absolute values = boundaries
        boundary_mask = torch.abs(edges)
        
        # Normalize to [0, 1] range
        boundary_mask = boundary_mask / (boundary_mask.max() + 1e-8)
        
        return boundary_mask
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, X, Y, Z) predicted occupancy logits
            target: (B, X, Y, Z) or (B, 1, X, Y, Z) ground truth bool/float
        
        Returns:
            loss: scalar
        """
        # Ensure target has channel dimension
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Convert target to float if boolean
        if target.dtype == torch.bool:
            target = target.float()
        
        # Detect boundaries
        boundary_mask = self._detect_boundaries(target)
        
        # Compute per-voxel BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        )
        
        # Apply boundary weighting
        # weights = 1.0 + boundary_weight * boundary_mask
        weights = 1.0 + (self.boundary_weight - 1.0) * boundary_mask
        weighted_loss = weights * bce_loss
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: BCE + Focal + Boundary weighting
    Allows flexible weighting of different loss components
    """
    def __init__(self, 
                 use_bce=True, bce_weight=1.0, bce_pos_weight=None,
                 use_focal=False, focal_weight=0.0, focal_alpha=0.25, focal_gamma=2.0,
                 use_boundary=False, boundary_weight_loss=0.0, boundary_weight_mult=5.0):
        super().__init__()
        
        self.use_bce = use_bce
        self.bce_weight = bce_weight
        
        self.use_focal = use_focal
        self.focal_weight = focal_weight
        
        self.use_boundary = use_boundary
        self.boundary_weight_loss = boundary_weight_loss
        
        if use_bce:
            self.bce_loss = BCEWithLogitsLoss3D(pos_weight=bce_pos_weight)
        
        if use_focal:
            self.focal_loss = FocalLoss3D(alpha=focal_alpha, gamma=focal_gamma)
        
        if use_boundary:
            self.boundary_loss = BoundaryWeightedLoss3D(boundary_weight=boundary_weight_mult)
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, X, Y, Z) predicted occupancy logits
            target: (B, X, Y, Z) or (B, 1, X, Y, Z) ground truth bool/float
        
        Returns:
            loss: scalar
            loss_dict: dict of individual loss components
        """
        total_loss = 0.0
        loss_dict = {}
        
        if self.use_bce:
            bce = self.bce_loss(logits, target)
            total_loss += self.bce_weight * bce
            loss_dict['bce'] = bce.item()
        
        if self.use_focal:
            focal = self.focal_loss(logits, target)
            total_loss += self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        
        if self.use_boundary:
            boundary = self.boundary_loss(logits, target)
            total_loss += self.boundary_weight_loss * boundary
            loss_dict['boundary'] = boundary.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create dummy data
    batch_size = 2
    shape = (50, 48, 48)  # Small volume for testing
    
    logits = torch.randn(batch_size, 1, *shape).to(device)
    target = (torch.rand(batch_size, *shape) > 0.5).to(device)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target occupancy: {target.float().mean()*100:.2f}%\n")
    
    # Test BCE
    print("1. BCE Loss:")
    bce_loss = BCEWithLogitsLoss3D()
    bce_loss = bce_loss.to(device)
    loss = bce_loss(logits, target)
    print(f"   Loss: {loss.item():.4f}\n")
    
    # Test Focal Loss
    print("2. Focal Loss:")
    focal_loss = FocalLoss3D(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss.to(device)
    loss = focal_loss(logits, target)
    print(f"   Loss: {loss.item():.4f}\n")
    
    # Test Boundary-Weighted Loss
    print("3. Boundary-Weighted Loss:")
    boundary_loss = BoundaryWeightedLoss3D(boundary_weight=5.0)
    boundary_loss = boundary_loss.to(device)
    loss = boundary_loss(logits, target)
    print(f"   Loss: {loss.item():.4f}\n")
    
    # Test Combined Loss
    print("4. Combined Loss (BCE + Focal):")
    combined_loss = CombinedLoss(
        use_bce=True, bce_weight=1.0,
        use_focal=True, focal_weight=0.5,
        use_boundary=False
    )
    combined_loss = combined_loss.to(device)
    loss, loss_dict = combined_loss(logits, target)
    print(f"   Total: {loss_dict['total']:.4f}")
    print(f"   BCE: {loss_dict['bce']:.4f}")
    print(f"   Focal: {loss_dict['focal']:.4f}\n")
    
    print("✓ All loss functions tested successfully!")


if __name__ == '__main__':
    test_losses()
