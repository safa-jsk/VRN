"""
PyTorch C++ Extension for Custom CUDA Marching Cubes
Uses pre-compiled extension (built via setup.py)
"""

import torch
import numpy as np
import os
import sys

# Import pre-compiled extension
try:
    import marching_cubes_cuda_ext
    CUDA_EXT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CUDA extension: {e}")
    print("  Run: python3 setup.py build_ext --inplace")
    CUDA_EXT_AVAILABLE = False


def marching_cubes_gpu(volume, isolevel=0.5, device='cuda'):
    """
    GPU-accelerated marching cubes using custom CUDA kernel
    
    Args:
        volume: torch.Tensor (D, H, W) - 3D voxel grid
        isolevel: float - isosurface threshold
        device: str - 'cuda' or 'cpu'
    
    Returns:
        vertices: torch.Tensor (N, 3) - mesh vertices
        faces: torch.Tensor (M, 3) - triangle indices
    """
    if not CUDA_EXT_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Run: python3 setup.py build_ext --inplace")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Ensure volume is on GPU
    if volume.device.type != 'cuda':
        volume = volume.to(device)
    
    # Get dimensions
    dimZ, dimY, dimX = volume.shape
    
    # Estimate max vertices/triangles (conservative)
    max_triangles = dimX * dimY * dimZ * 5
    max_vertices = max_triangles * 3
    
    # Allocate output buffers on GPU
    vertices = torch.zeros((max_vertices, 3), dtype=torch.float32, device=device)
    triangles = torch.zeros((max_triangles, 3), dtype=torch.int32, device=device)
    num_vertices = torch.zeros(1, dtype=torch.int32, device=device)
    num_triangles = torch.zeros(1, dtype=torch.int32, device=device)
    
    # Call CUDA kernel
    marching_cubes_cuda_ext.marching_cubes_forward(
        volume,
        vertices,
        triangles,
        num_vertices,
        num_triangles,
        float(isolevel),
        int(dimX), int(dimY), int(dimZ),
        int(max_vertices), int(max_triangles)
    )
    
    # Get actual counts
    nv = num_vertices.item()
    nt = num_triangles.item()
    
    # Trim to actual size
    vertices = vertices[:nv, :]
    faces = triangles[:nt, :]
    
    return vertices, faces


class MarchingCubesCUDA(torch.nn.Module):
    """
    PyTorch Module wrapper for CUDA marching cubes
    Enables integration into neural network pipelines
    """
    def __init__(self, isolevel=0.5):
        super().__init__()
        self.isolevel = isolevel
        self.cuda_ext = None
    
    def forward(self, volume):
        """
        Args:
            volume: (B, D, H, W) or (D, H, W) tensor
        
        Returns:
            List of (vertices, faces) tuples for each batch
        """
        if volume.dim() == 3:
            # Single volume
            return marching_cubes_gpu(volume, self.isolevel)
        elif volume.dim() == 4:
            # Batched volumes
            results = []
            for i in range(volume.shape[0]):
                verts, faces = marching_cubes_gpu(volume[i], self.isolevel)
                results.append((verts, faces))
            return results
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {volume.dim()}D")


if __name__ == '__main__':
    # Test CUDA extension
    print("Testing CUDA Marching Cubes Extension...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create test volume (sphere)
        size = 64
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        z = torch.linspace(-1, 1, size)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        volume = torch.sqrt(X**2 + Y**2 + Z**2)
        volume = volume.to(device)
        
        print(f"Test volume shape: {volume.shape}")
        
        # Run marching cubes
        try:
            verts, faces = marching_cubes_gpu(volume, isolevel=0.5)
            print(f"✓ Generated mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print("✗ CUDA not available")
