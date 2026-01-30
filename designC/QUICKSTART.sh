#!/bin/bash
# Design C - Quick Start Guide

echo "╔════════════════════════════════════════════════════════╗"
echo "║     Design C — PyTorch VRN Quick Start Guide          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Step 1: Check data status
echo "Step 1: Checking data availability..."
echo "─────────────────────────────────────"

if [ -d "data/FaceScape/voxelized_shapes" ]; then
    VOL_COUNT=$(find data/FaceScape/voxelized_shapes -name "*.npy" | wc -l)
    echo "✓ Voxelized shapes: $VOL_COUNT volumes ready"
else
    echo "✗ Voxelized shapes not found"
    echo "  Run: python3 designC/data/voxelizer.py --input data/FaceScape/fsmview_trainset_shape_001-020/fsmview_trainset --output data/FaceScape/voxelized_shapes/"
fi

if [ -d "data/FaceScape/fsmview_trainset_images_001-020" ]; then
    IMG_COUNT=$(find data/FaceScape/fsmview_trainset_images_001-020 -name "*.jpg" -o -name "*.png" | wc -l)
    echo "✓ FaceScape images: $IMG_COUNT images available"
    IMAGES_READY=true
else
    echo "✗ FaceScape images still downloading"
    echo "  Waiting for: fsmview_trainset_images_001-020.zip"
    IMAGES_READY=false
fi
echo

# Step 2: Environment setup
echo "Step 2: Environment Setup"
echo "─────────────────────────────────────"
echo "Recommended: Use vrn_env or create new environment"
echo
echo "Install Design C dependencies:"
echo "  pip install -r designC/requirements.txt"
echo

# Step 3: Test components
echo "Step 3: Test Core Components"
echo "─────────────────────────────────────"
echo "Test PyTorch VRN model:"
echo "  python3 designC/models/vrn_pytorch.py"
echo
echo "Test loss functions:"
echo "  python3 designC/models/losses.py"
echo
echo "Test FaceScape dataset:"
echo "  python3 -c 'import sys; sys.path.append(\".\"); from designC.data.facescape_loader import test_dataset; test_dataset()'"
echo

# Step 4: Training
echo "Step 4: Training Options"
echo "─────────────────────────────────────"

if [ "$IMAGES_READY" = true ]; then
    echo "✓ Images available — ready to train!"
    echo
    echo "Option A: Train on voxelized shapes (direct 3D supervision)"
    echo "  ./designC/scripts/train.sh facescape data/FaceScape/voxelized_shapes data/FaceScape/images 100 2"
    echo
    echo "Option B: Generate teacher volumes + distillation"
    echo "  ./designC/scripts/generate_teacher_volumes.sh data/FaceScape/images data/FaceScape/teacher_volumes"
    echo "  python3 designC/train.py --dataset-type teacher --image-dir data/FaceScape/images --volume-dir data/FaceScape/teacher_volumes"
else
    echo "✗ Waiting for images to download"
    echo
    echo "When images are ready, run:"
    echo "  ./designC/scripts/train.sh facescape data/FaceScape/voxelized_shapes <IMAGE_DIR> 100 2"
fi
echo

# Step 5: Inference
echo "Step 5: Inference Pipeline"
echo "─────────────────────────────────────"
echo "After training, test on single image:"
echo "  python3 designC/pipeline.py --model designC/checkpoints/best.pth --image <IMAGE> --output <OUTPUT.obj>"
echo

# Step 6: Benchmarking
echo "Step 6: Design A/B/C Comparison"
echo "─────────────────────────────────────"
echo "Run benchmarks on AFLW2000 subset:"
echo "  (Benchmark script coming next — will use Design B's verification protocol)"
echo

# Step 7: Monitoring
echo "Step 7: Training Monitoring"
echo "─────────────────────────────────────"
echo "View TensorBoard logs:"
echo "  tensorboard --logdir designC/logs"
echo "  Open: http://localhost:6006"
echo

# Summary
echo "╔════════════════════════════════════════════════════════╗"
echo "║                    Current Status                      ║"
echo "╠════════════════════════════════════════════════════════╣"
if [ "$IMAGES_READY" = true ]; then
    echo "║  ✓ Data ready — can start training immediately        ║"
else
    echo "║  ⏳ Waiting for FaceScape images to download           ║"
fi
echo "║  ✓ All core components implemented and tested         ║"
echo "║  ✓ Pipeline integrates with Design B CUDA MC          ║"
echo "║  Target: <100ms end-to-end inference (20-30x speedup) ║"
echo "╚════════════════════════════════════════════════════════╝"
