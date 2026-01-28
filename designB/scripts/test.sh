#!/bin/bash
# Test Custom CUDA Implementation with Design B Pipeline

set -e

echo "=========================================="
echo "Testing Custom CUDA Marching Cubes"
echo "=========================================="
echo ""

# Activate environment
source vrn_env/bin/activate

# Check if extension is built
if [ ! -f "marching_cubes_cuda_ext.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "✗ CUDA extension not found!"
    echo "  Building now..."
    python3 setup.py build_ext --inplace
    echo ""
fi

# Test on first volume
VOL="data/out/designB/volumes/image00002.npy"

if [ ! -f "$VOL" ]; then
    echo "✗ Test volume not found: $VOL"
    echo "  Run Design B pipeline first: ./scripts/designB_run.sh"
    exit 1
fi

echo "Testing on Design B volume: $VOL"
echo ""

# Run single volume test
cd cuda_post
python3 << EOF
import sys
sys.path.insert(0, '..')

from volume_io import load_volume_npy
from marching_cubes_cuda import marching_cubes_gpu_pytorch
import time

# Load volume
print("Loading volume...")
volume = load_volume_npy('$VOL')
print(f"  Volume shape: {volume.shape}")

# Run GPU marching cubes
print("\nRunning custom CUDA marching cubes...")
t0 = time.time()
verts, faces = marching_cubes_gpu_pytorch(volume, threshold=10.0)
t_gpu = time.time() - t0

print(f"  Time: {t_gpu:.4f}s")
print(f"  Mesh: {len(verts)} vertices, {len(faces)} faces")
print(f"\n✓ Custom CUDA kernel working on Design B volumes!")
EOF

echo ""
echo "=========================================="
echo "✓ Test Passed"
echo "=========================================="
