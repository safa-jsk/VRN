#!/bin/bash
# Design B - Two-Stage Pipeline
# Stage 1: VRN volume extraction (CPU)
# Stage 2: GPU marching cubes (CUDA)

set -e

# Configuration
INPUT_DIR="${1:-data/in/aflw2000}"
OUTPUT_DIR="${2:-data/out/designB}"
USE_GPU="${3:-true}"

VOLUME_DIR="$OUTPUT_DIR/volumes"
MESH_DIR="$OUTPUT_DIR/meshes"
LOGS_DIR="$OUTPUT_DIR/logs"
DESIGN_A_DIR="data/out/designA"  # Source for demo volumes

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "======================================================================"
echo "Design B - CUDA-Accelerated VRN Pipeline"
echo "======================================================================"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPU enabled: $USE_GPU"
echo ""

# Create output directories
mkdir -p "$VOLUME_DIR" "$MESH_DIR" "$LOGS_DIR"

# Check CUDA availability
if [ "$USE_GPU" = "true" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU detected:${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo ""
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found, GPU may not be available${NC}"
        echo ""
    fi
fi

# Stage 1: Volume Extraction
echo "======================================================================"
echo -e "${BLUE}Stage 1: Volume Generation (Demo Mode)${NC}"
echo "======================================================================"
echo -e "${YELLOW}NOTE: Using Design A meshes to create demo volumes${NC}"
echo -e "${YELLOW}      (VRN Docker doesn't export volumes by default)${NC}"
echo ""

START_STAGE1=$(date +%s)

# Use Design A meshes to create demo volumes for GPU demonstration
python3 cuda_post/create_demo_volumes.py \
    --designA "$DESIGN_A_DIR" \
    --output "$VOLUME_DIR" | tee "$LOGS_DIR/stage1_volumes.log"

END_STAGE1=$(date +%s)
STAGE1_TIME=$((END_STAGE1 - START_STAGE1))

echo ""

# Stage 2: GPU Marching Cubes
echo "======================================================================"
echo -e "${BLUE}Stage 2: GPU Marching Cubes (CUDA)${NC}"
echo "======================================================================"

START_STAGE2=$(date +%s)

# Build GPU flag
GPU_FLAG=""
if [ "$USE_GPU" = "false" ]; then
    GPU_FLAG="--cpu"
fi

# Run GPU marching cubes on all volumes
python3 cuda_post/marching_cubes_cuda.py \
    --input "$VOLUME_DIR" \
    --output "$MESH_DIR" \
    --threshold 10.0 \
    $GPU_FLAG \
    | tee "$LOGS_DIR/stage2_marching_cubes.log"

END_STAGE2=$(date +%s)
STAGE2_TIME=$((END_STAGE2 - START_STAGE2))

# Overall summary
echo ""
echo "======================================================================"
echo "Design B Pipeline - Complete"
echo "======================================================================"
echo "Stage 1 (Volume extraction): ${STAGE1_TIME}s"
echo "Stage 2 (GPU marching cubes): ${STAGE2_TIME}s"
echo "Total time: $((STAGE1_TIME + STAGE2_TIME))s"
echo ""
echo "Output directories:"
echo "  Volumes: $VOLUME_DIR"
echo "  Meshes:  $MESH_DIR"
echo "  Logs:    $LOGS_DIR"
echo ""

# Count outputs
VOLUME_COUNT=$(ls -1 "$VOLUME_DIR"/*.npy 2>/dev/null | wc -l)
MESH_COUNT=$(ls -1 "$MESH_DIR"/*.obj 2>/dev/null | wc -l)

echo "Generated outputs:"
echo "  Volumes: $VOLUME_COUNT"
echo "  Meshes:  $MESH_COUNT"
echo "======================================================================"

# Suggest next steps
echo ""
echo "Next steps:"
echo "  1. Verify meshes: bash scripts/designB_verify.sh"
echo "  2. Generate comparisons: python3 cuda_post/generate_poster_figures.py"
echo "  3. View results in: $OUTPUT_DIR"
