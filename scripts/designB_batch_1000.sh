#!/bin/bash
set -euo pipefail

# DesignB – Batch pipeline + benchmark on 468 volumes from DesignA_GPU
# Uses Docker image vrn/designb:latest (Python 3.12, PyTorch cu120, custom CUDA MC)
#
# Pipeline:
#   Step 1: Convert .raw → .npy  (needed by benchmark.py)
#   Step 2: GPU MC pipeline      (src.designB.pipeline) → .obj meshes
#   Step 3: CPU vs GPU benchmark (src.designB.benchmark) → timing JSON + plots
#
# Usage: bash scripts/designB_batch_1000.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DOCKER_IMAGE="vrn/designb:latest"
RAW_INPUT_DIR="artifacts/designA_gpu/raw"   # 468 .raw volumes from DesignA_GPU
NPY_DIR="artifacts/designB/volumes"         # Converted .npy volumes
OBJ_DIR="artifacts/designB/obj"             # GPU MC mesh output
BENCH_DIR="artifacts/designB/benchmarks"   # Benchmark results
LOG_FILE="artifacts/designB/batch.log"

# Docker run helper – mounts project, enables GPU
docker_run() {
    docker run --rm \
        --runtime=nvidia \
        --gpus all \
        -v "${PROJECT_DIR}:/workspace" \
        -w /workspace \
        "$DOCKER_IMAGE" \
        "$@"
}

# ── Prerequisites ──────────────────────────────────────────────────

if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "ERROR: Docker image not found: $DOCKER_IMAGE"
    echo "Run: docker build -t $DOCKER_IMAGE -f DesignB/Dockerfile ."
    exit 1
fi

if [ ! -d "$RAW_INPUT_DIR" ] || [ -z "$(ls -A "$RAW_INPUT_DIR"/*.raw 2>/dev/null)" ]; then
    echo "ERROR: No .raw files found in $RAW_INPUT_DIR"
    echo "Run DesignA_GPU first: bash scripts/designA_gpu_batch_1000.sh"
    exit 1
fi

mkdir -p "$NPY_DIR" "$OBJ_DIR" "$BENCH_DIR" "$(dirname "$LOG_FILE")"
raw_count=$(ls "$RAW_INPUT_DIR"/*.raw | wc -l)

echo "============================================" | tee "$LOG_FILE"
echo "DesignB Batch — $(date)" | tee -a "$LOG_FILE"
echo "Input volumes (.raw): $raw_count" | tee -a "$LOG_FILE"
echo "Image: $DOCKER_IMAGE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Verify GPU access in container
echo "" | tee -a "$LOG_FILE"
echo "GPU check:" | tee -a "$LOG_FILE"
docker_run python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  ERROR: no CUDA'); exit(1)
" 2>&1 | tee -a "$LOG_FILE"

# Verify CUDA extension loads in container
echo "" | tee -a "$LOG_FILE"
echo "CUDA extension check:" | tee -a "$LOG_FILE"
docker_run python3 -c "
import sys, os
# Mirror what ensure_ext_importable() does
for p in ['external/marching_cubes_cuda_ext', 'artifacts/build']:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
from external.marching_cubes_cuda_ext.cuda_marching_cubes import marching_cubes_gpu
import torch
v = torch.rand(20,19,19,dtype=torch.float32).cuda()
vt,ft = marching_cubes_gpu(v, isolevel=0.5, device='cuda')
print(f'  OK — marching_cubes_gpu: {vt.shape[0]} verts')
" 2>&1 | tee -a "$LOG_FILE"

# ── Step 1: Convert .raw → .npy ────────────────────────────────────

npy_count=$(find "$NPY_DIR" -maxdepth 1 -name "*.npy" 2>/dev/null | wc -l)
echo "" | tee -a "$LOG_FILE"
echo "Step 1: Convert .raw → .npy (already done: $npy_count)" | tee -a "$LOG_FILE"

if [ "$npy_count" -lt "$raw_count" ]; then
    echo "  Converting ${raw_count} volumes…" | tee -a "$LOG_FILE"
    step1_start=$(date +%s)

    docker_run python3 -m src.designB.convert_raw_to_npy \
        --input  "$RAW_INPUT_DIR" \
        --output "$NPY_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    echo "  Step 1 done ($(( $(date +%s) - step1_start ))s)" | tee -a "$LOG_FILE"
else
    echo "  All volumes already converted, skipping." | tee -a "$LOG_FILE"
fi

npy_count=$(find "$NPY_DIR" -maxdepth 1 -name "*.npy" 2>/dev/null | wc -l)
echo "  .npy files: $npy_count" | tee -a "$LOG_FILE"

# ── Step 2: GPU MC pipeline → .obj meshes ──────────────────────────

obj_count=$(find "$OBJ_DIR" -maxdepth 1 -name "*.obj" 2>/dev/null | wc -l)
echo "" | tee -a "$LOG_FILE"
echo "Step 2: GPU MC pipeline → meshes (already done: $obj_count)" | tee -a "$LOG_FILE"

if [ "$obj_count" -lt "$npy_count" ]; then
    step2_start=$(date +%s)

    docker_run python3 -m src.designB.pipeline \
        --input     "$NPY_DIR" \
        --output    "$OBJ_DIR" \
        --pattern   "*.npy" \
        --threshold 0.5 \
        --cudnn-benchmark true \
        --tf32 true \
        --warmup-iters 15 \
        2>&1 | tee -a "$LOG_FILE"

    echo "  Step 2 done ($(( $(date +%s) - step2_start ))s)" | tee -a "$LOG_FILE"
else
    echo "  All meshes already generated, skipping." | tee -a "$LOG_FILE"
fi

# ── Step 3: CPU vs GPU benchmark ───────────────────────────────────

echo "" | tee -a "$LOG_FILE"
echo "Step 3: CPU vs GPU benchmark" | tee -a "$LOG_FILE"
step3_start=$(date +%s)

docker_run python3 -m src.designB.benchmark \
    --volumes  "$NPY_DIR" \
    --output   "$BENCH_DIR" \
    --runs     3 \
    --warmup-iters 15 \
    --cudnn-benchmark true \
    --tf32 true \
    --plot \
    2>&1 | tee -a "$LOG_FILE"

echo "  Step 3 done ($(( $(date +%s) - step3_start ))s)" | tee -a "$LOG_FILE"

# ── Summary ────────────────────────────────────────────────────────

obj_count=$(find "$OBJ_DIR" -maxdepth 1 -name "*.obj" 2>/dev/null | wc -l)
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "COMPLETE — $(date)" | tee -a "$LOG_FILE"
echo "  .raw input:   $raw_count" | tee -a "$LOG_FILE"
echo "  .npy volumes: $npy_count" | tee -a "$LOG_FILE"
echo "  .obj meshes:  $obj_count" | tee -a "$LOG_FILE"
echo "  Benchmark:    $BENCH_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
