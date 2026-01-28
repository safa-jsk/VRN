#!/usr/bin/env python3
"""
Test Custom CUDA Marching Cubes Kernel
Quick validation that GPU acceleration is working
"""

import torch
import numpy as np
import time
from cuda_kernels.cuda_marching_cubes import marching_cubes_gpu
from skimage.measure import marching_cubes as marching_cubes_cpu

def create_sphere_volume(size=64, radius=0.4):
    """Create a sphere volume for testing"""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    volume = torch.sqrt(X**2 + Y**2 + Z**2)
    
    return volume

print("="*60)
print("Testing Custom CUDA Marching Cubes Kernel")
print("="*60)

# Check GPU
if not torch.cuda.is_available():
    print("✗ CUDA not available!")
    exit(1)

device = torch.device('cuda')
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print()

# Create test volume
size = 128
print(f"Creating test volume ({size}³ voxels)...")
volume = create_sphere_volume(size)
print(f"  Volume shape: {volume.shape}")
print()

# Test CPU baseline
print("Running CPU marching cubes (scikit-image)...")
volume_cpu = volume.numpy()
t0 = time.time()
verts_cpu, faces_cpu, normals, values = marching_cubes_cpu(volume_cpu, level=0.5)
t_cpu = time.time() - t0
print(f"  CPU Time: {t_cpu:.4f}s")
print(f"  Mesh: {verts_cpu.shape[0]} vertices, {faces_cpu.shape[0]} faces")
print()

# Test GPU CUDA kernel
print("Running GPU marching cubes (custom CUDA kernel)...")
volume_gpu = volume.to(device)
t0 = time.time()
try:
    verts_gpu, faces_gpu = marching_cubes_gpu(volume_gpu, isolevel=0.5, device='cuda')
    t_gpu = time.time() - t0
    print(f"  GPU Time: {t_gpu:.4f}s")
    print(f"  Mesh: {verts_gpu.shape[0]} vertices, {faces_gpu.shape[0]} faces")
    print()
    
    # Calculate speedup
    speedup = t_cpu / t_gpu
    print("="*60)
    print(f"SPEEDUP: {speedup:.2f}x")
    print("="*60)
    
    if speedup > 1.0:
        print(f"✓ GPU is {speedup:.2f}x FASTER than CPU")
    else:
        print(f"⚠ GPU is {1/speedup:.2f}x SLOWER than CPU (likely transfer overhead)")
    
except Exception as e:
    print(f"✗ GPU marching cubes failed: {e}")
    import traceback
    traceback.print_exc()
