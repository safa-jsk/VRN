#!/bin/bash
set -euo pipefail

# DesignA_GPU — Batch process 1000 images from 300W_LP/AFW
# Pipeline:
#   1. Docker VRN inference (Torch7 --device gpu) → .raw volumes
#   2. Host GPU marching cubes (CUDA kernel, PyTorch) → .obj meshes
# Usage: bash scripts/designA_gpu_batch_1000.sh [path_list]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PATHS_FILE="${1:-docs/ops/300w_lp_afw_1000.txt}"
DATASET_DIR="/home/T2520734/Downloads/300W-LP/300W_LP"
OUTPUT_DIR="artifacts/designA_gpu"
RAW_DIR="${OUTPUT_DIR}/raw"
OBJ_DIR="${OUTPUT_DIR}/obj"
CROP_DIR="${OUTPUT_DIR}/crop"
LOG_FILE="${OUTPUT_DIR}/batch.log"
TIME_LOG="${OUTPUT_DIR}/timing.csv"
DOCKER_IMAGE="asjackson/vrn:latest"
RUNNER_SCRIPT="${PROJECT_DIR}/DesignA_GPU/scripts/docker_run_raw.sh"
PYTHON="/home/T2520734/Documents/music-generation-unsupervised/venv/bin/python3"

# ── Prerequisites ──────────────────────────────────────────────────

if [ ! -f "$PATHS_FILE" ]; then
    echo "ERROR: Path list not found: $PATHS_FILE"
    exit 1
fi

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found: $DATASET_DIR"
    exit 1
fi

if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
    echo "ERROR: Docker image not found: $DOCKER_IMAGE"
    exit 1
fi

if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "ERROR: GPU runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: Python GPU environment not available: $PYTHON"
    exit 1
fi

# ── Setup ──────────────────────────────────────────────────────────

mkdir -p "$RAW_DIR" "$OBJ_DIR" "$CROP_DIR"

mapfile -t images < "$PATHS_FILE"
total=${#images[@]}

echo "============================================" | tee "$LOG_FILE"
echo "DesignA_GPU Batch — $(date)" | tee -a "$LOG_FILE"
echo "Images:  $total" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_DIR" | tee -a "$LOG_FILE"
echo "Output:  $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "GPU:     $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

if [ ! -f "$TIME_LOG" ]; then
    echo "index,filename,status,docker_sec,mc_sec,total_sec" > "$TIME_LOG"
fi

# ── Phase 1: Docker VRN Inference ──────────────────────────────────

echo "" | tee -a "$LOG_FILE"
echo "Phase 1: VRN inference (Docker GPU)" | tee -a "$LOG_FILE"
echo "--------------------------------------------" | tee -a "$LOG_FILE"

phase1_start=$(date +%s)
success_docker=0
fail_docker=0
skip_docker=0

for i in "${!images[@]}"; do
    img_rel="${images[$i]}"
    [ -z "$img_rel" ] && continue

    img_name=$(basename "$img_rel")
    count=$((i + 1))
    raw_out="${RAW_DIR}/${img_name}.raw"

    # Resumable: skip if .raw (or final .obj) already produced
    if [ -f "$raw_out" ] || [ -f "${OBJ_DIR}/${img_name}.obj" ]; then
        skip_docker=$((skip_docker + 1))
        printf "[%4d/%d] SKIP %s\n" "$count" "$total" "$img_name"
        continue
    fi

    # Path inside the dataset mount (/data/300W_LP/...)
    container_path="/data/300W_LP/${img_rel#data/300W_LP/}"
    src_dir="${DATASET_DIR}/$(dirname "${img_rel#data/300W_LP/}")"

    printf "[%4d/%d] %s … " "$count" "$total" "$img_name"

    start_s=$(date +%s)
    docker_out=$(mktemp)

    # Run VRN in Docker (CPU inference; GPU MC runs on host after)
    if docker run --rm \
        -v "${DATASET_DIR}:/data/300W_LP" \
        -v "${RUNNER_SCRIPT}:/runner/run_raw.sh:ro" \
        --entrypoint /runner/run_raw.sh \
        "$DOCKER_IMAGE" /runner/run_raw.sh "$container_path" > "$docker_out" 2>&1; then

        cat "$docker_out" >> "$LOG_FILE"

        raw_src="${src_dir}/${img_name}.raw"
        crop_src="${src_dir}/${img_name}.crop.jpg"

        if [ -f "$raw_src" ]; then
            mv "$raw_src" "$raw_out"
            [ -f "$crop_src" ] && mv "$crop_src" "${CROP_DIR}/${img_name}.crop.jpg"
            elapsed=$(($(date +%s) - start_s))
            printf "RAW OK (%ds)\n" "$elapsed"
            success_docker=$((success_docker + 1))
        else
            elapsed=$(($(date +%s) - start_s))
            printf "FAIL (no .raw)\n"
            cat "$docker_out" >> "$LOG_FILE"
            fail_docker=$((fail_docker + 1))
        fi
    else
        cat "$docker_out" >> "$LOG_FILE"
        elapsed=$(($(date +%s) - start_s))

        if grep -q "face detector failed" "$docker_out"; then
            printf "FAIL (no face detected)\n"
        else
            printf "FAIL (error)\n"
        fi
        fail_docker=$((fail_docker + 1))
    fi
    rm -f "$docker_out"
done

phase1_sec=$(($(date +%s) - phase1_start))
echo "" | tee -a "$LOG_FILE"
echo "Phase 1 complete: ${success_docker} ok, ${fail_docker} fail, ${skip_docker} skip (${phase1_sec}s)" | tee -a "$LOG_FILE"

# ── Phase 2: GPU Marching Cubes ────────────────────────────────────

raw_count=$(ls "$RAW_DIR"/*.raw 2>/dev/null | wc -l)
already_done=$(ls "$OBJ_DIR"/*.obj 2>/dev/null | wc -l)

echo "" | tee -a "$LOG_FILE"
echo "Phase 2: GPU marching cubes (CUDA kernel)" | tee -a "$LOG_FILE"
echo "  Volumes to process: $raw_count" | tee -a "$LOG_FILE"
echo "  Already meshed:     $already_done" | tee -a "$LOG_FILE"
echo "--------------------------------------------" | tee -a "$LOG_FILE"

if [ "$raw_count" -gt 0 ]; then
    phase2_start=$(date +%s)

    "$PYTHON" -m src.designB.pipeline \
        --input  "$RAW_DIR" \
        --output "$OBJ_DIR" \
        --pattern "*.raw" \
        --threshold 0.5 \
        --cudnn-benchmark true \
        --tf32 true \
        --warmup-iters 15 \
        2>&1 | tee -a "$LOG_FILE"

    phase2_sec=$(($(date +%s) - phase2_start))
    echo "" | tee -a "$LOG_FILE"
    echo "Phase 2 complete (${phase2_sec}s)" | tee -a "$LOG_FILE"
else
    echo "No .raw files to process — skipping Phase 2" | tee -a "$LOG_FILE"
    phase2_sec=0
fi

# ── Summary ────────────────────────────────────────────────────────

obj_count=$(ls "$OBJ_DIR"/*.obj 2>/dev/null | wc -l)
total_sec=$((phase1_sec + phase2_sec))
total_min=$((total_sec / 60))
total_rem=$((total_sec % 60))

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "COMPLETE — $(date)" | tee -a "$LOG_FILE"
echo "  Images:        $total" | tee -a "$LOG_FILE"
echo "  Docker ok:     $success_docker  fail: $fail_docker  skip: $skip_docker" | tee -a "$LOG_FILE"
echo "  Meshes (.obj): $obj_count" | tee -a "$LOG_FILE"
echo "  Phase 1 (VRN inference): ${phase1_sec}s" | tee -a "$LOG_FILE"
echo "  Phase 2 (GPU MC):        ${phase2_sec}s" | tee -a "$LOG_FILE"
echo "  Wall time: ${total_min}m ${total_rem}s" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Write per-image CSV from pipeline MC timing log if it exists
mc_timing="${OBJ_DIR}/marching_cubes_timing.log"
if [ -f "$mc_timing" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Per-volume MC timing: $mc_timing" | tee -a "$LOG_FILE"
fi
