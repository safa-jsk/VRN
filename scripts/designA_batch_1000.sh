#!/bin/bash
set -euo pipefail

# DesignA_CPU — Batch process 1000 images from 300W_LP/AFW via Docker
# Usage: bash scripts/designA_batch_1000.sh [path_list]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PATHS_FILE="${1:-docs/ops/300w_lp_afw_1000.txt}"
DATASET_DIR="/home/T2520734/Downloads/300W-LP/300W_LP"
OUTPUT_DIR="artifacts/designA_cpu"
OBJ_DIR="${OUTPUT_DIR}/obj"
CROP_DIR="${OUTPUT_DIR}/crop"
LOG_FILE="${OUTPUT_DIR}/batch.log"
TIME_LOG="${OUTPUT_DIR}/timing.csv"
DOCKER_IMAGE="asjackson/vrn:latest"

# Verify prerequisites
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

# Create output directories
mkdir -p "$OBJ_DIR" "$CROP_DIR"

# Load image list
mapfile -t images < "$PATHS_FILE"
total=${#images[@]}
echo "============================================" | tee "$LOG_FILE"
echo "DesignA_CPU Batch — $(date)" | tee -a "$LOG_FILE"
echo "Images: $total" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_DIR" | tee -a "$LOG_FILE"
echo "Output:  $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# CSV header
if [ ! -f "$TIME_LOG" ]; then
    echo "index,filename,status,elapsed_sec" > "$TIME_LOG"
fi

batch_start=$(date +%s)
success=0
fail=0
skip=0

for i in "${!images[@]}"; do
    img_rel="${images[$i]}"
    [ -z "$img_rel" ] && continue

    img_name=$(basename "$img_rel")
    count=$((i + 1))

    # Resumable: skip if .obj already exists in output
    if [ -f "${OBJ_DIR}/${img_name}.obj" ]; then
        skip=$((skip + 1))
        printf "[%4d/%d] SKIP %s\n" "$count" "$total" "$img_name"
        continue
    fi

    # Path inside dataset (strip data/ prefix from list)
    container_path="/data/${img_rel#data/}"

    printf "[%4d/%d] %s … " "$count" "$total" "$img_name"

    start_s=$(date +%s)
    docker_out=$(mktemp)

    # Run VRN in Docker
    if docker run --rm \
        -v "$PWD/data:/data" \
        -v "${DATASET_DIR}:/data/300W_LP" \
        "$DOCKER_IMAGE" /runner/run.sh "$container_path" > "$docker_out" 2>&1; then

        cat "$docker_out" >> "$LOG_FILE"

        # Locate outputs (written next to source in the dataset mount)
        src_dir="${DATASET_DIR}/$(dirname "${img_rel#data/300W_LP/}")"
        obj_file="${src_dir}/${img_name}.obj"
        crop_file="${src_dir}/${img_name}.crop.jpg"

        if [ -f "$obj_file" ]; then
            mv "$obj_file" "${OBJ_DIR}/"
            [ -f "$crop_file" ] && mv "$crop_file" "${CROP_DIR}/"
            elapsed=$(($(date +%s) - start_s))
            echo "${count},${img_name},ok,${elapsed}" >> "$TIME_LOG"
            success=$((success + 1))
            printf "OK (%ds)\n" "$elapsed"
        else
            elapsed=$(($(date +%s) - start_s))
            echo "${count},${img_name},no_output,${elapsed}" >> "$TIME_LOG"
            fail=$((fail + 1))
            [ -f "$crop_file" ] && rm -f "$crop_file"
            printf "FAIL (no .obj)\n"
        fi
    else
        cat "$docker_out" >> "$LOG_FILE"
        elapsed=$(($(date +%s) - start_s))

        # Classify error
        if grep -q "face detector failed" "$docker_out"; then
            echo "${count},${img_name},no_face,${elapsed}" >> "$TIME_LOG"
            fail=$((fail + 1))
            printf "FAIL (no face detected)\n"
        else
            echo "${count},${img_name},error,${elapsed}" >> "$TIME_LOG"
            fail=$((fail + 1))
            printf "FAIL (error)\n"
        fi

        # Clean up any partial outputs
        src_dir="${DATASET_DIR}/$(dirname "${img_rel#data/300W_LP/}")"
        rm -f "${src_dir}/${img_name}.obj" "${src_dir}/${img_name}.crop.jpg"
    fi
    rm -f "$docker_out"
done

total_sec=$(($(date +%s) - batch_start))
total_min=$((total_sec / 60))
total_rem=$((total_sec % 60))

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "COMPLETE — $(date)" | tee -a "$LOG_FILE"
echo "  Total:   $total" | tee -a "$LOG_FILE"
echo "  Success: $success" | tee -a "$LOG_FILE"
echo "  Failed:  $fail" | tee -a "$LOG_FILE"
echo "  Skipped: $skip" | tee -a "$LOG_FILE"
if [ $((success + fail)) -gt 0 ]; then
    rate=$(( success * 100 / (success + fail) ))
    echo "  Success rate: ${rate}%" | tee -a "$LOG_FILE"
fi
echo "  Wall time: ${total_min}m ${total_rem}s" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
