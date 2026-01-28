#!/bin/bash
# Design B - Complete Pipeline (CORRECT VERSION)
# Processes raw AFLW2000 images through VRN → CUDA Marching Cubes

set -e

echo "=========================================="
echo "Design B Pipeline - Correct Implementation"
echo "=========================================="
echo ""

# Stage 1: Extract volumes from VRN using raw images
echo "STAGE 1: Extracting volumes from VRN..."
echo "----------------------------------------"
./scripts/designB_stage1_extract_volumes_correct.sh

echo ""
echo ""

# Stage 2: GPU-accelerated marching cubes
echo "STAGE 2: GPU Marching Cubes (Custom CUDA)"
echo "----------------------------------------"

# Activate Python environment
source vrn_env/bin/activate

# Ensure CUDA extension is built
if [ ! -f "marching_cubes_cuda_ext.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "Building CUDA extension..."
    python3 setup.py build_ext --inplace
fi

# Run GPU marching cubes on extracted volumes
cd cuda_post
python3 marching_cubes_cuda.py \
    --volume-dir ../data/out/designB/volumes \
    --output-dir ../data/out/designB/meshes \
    --threshold 10.0 \
    --device cuda

cd ..

echo ""
echo "=========================================="
echo "Design B Pipeline Complete"
echo "=========================================="

# Summary
VOLUMES=$(find data/out/designB/volumes -name "*.npy" 2>/dev/null | wc -l)
MESHES=$(find data/out/designB/meshes -name "*.obj" 2>/dev/null | wc -l)

echo ""
echo "Results:"
echo "  Input:    AFLW2000 raw images"
echo "  Volumes:  $VOLUMES (.npy files from VRN)"
echo "  Meshes:   $MESHES (.obj files from CUDA MC)"
echo ""
echo "Output locations:"
echo "  Volumes: data/out/designB/volumes/"
echo "  Meshes:  data/out/designB/meshes/"
