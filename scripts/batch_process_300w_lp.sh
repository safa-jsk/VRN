#!/bin/bash
# Design A - Batch Processing Script for 300W_LP Dataset
# This script efficiently processes 300W_LP images and outputs to separate directory

INPUT_DIR="data/300W_LP"
OUTPUT_DIR="data/out/designA_300w_lp"
LOG_FILE="${OUTPUT_DIR}/batch_process.log"
TIME_LOG="${OUTPUT_DIR}/time.log"
SUBSET_FILE="${1:-}"

mkdir -p "${OUTPUT_DIR}"

echo "Design A - 300W_LP Batch Processing Started: $(date)" | tee "${LOG_FILE}"
echo "# Timing Log - Format: seconds filename" > "${TIME_LOG}"

# Build image paths from subset
declare -a images

if [ -n "$SUBSET_FILE" ]; then
    echo "Building full paths from subset file..." | tee -a "${LOG_FILE}"
    
    # Pre-scan all images in 300W_LP once
    declare -A img_map
    while IFS= read -r fullpath; do
        basename=$(basename "$fullpath")
        img_map["$basename"]="$fullpath"
    done < <(find "$INPUT_DIR" -maxdepth 2 -type f -name "*.jpg" 2>/dev/null)
    
    echo "Built index of ${#img_map[@]} images" | tee -a "${LOG_FILE}"
    
    # Match subset file to indexed images
    while IFS= read -r img_name; do
        if [ -n "${img_map[$img_name]}" ]; then
            images+=("${img_map[$img_name]}")
        fi
    done < "$SUBSET_FILE"
    
    echo "Matched ${#images[@]} images from subset" | tee -a "${LOG_FILE}"
fi

if [ ${#images[@]} -eq 0 ]; then
    echo "ERROR: No images found" | tee -a "${LOG_FILE}"
    exit 1
fi

batch_start=$(date +%s)
processed=0
successful=0
failed=0

echo "Starting batch: ${#images[@]} images" | tee -a "${LOG_FILE}"

for img_path in "${images[@]}"; do
    img_name=$(basename "$img_path")
    rel_path=${img_path#data/}  # Remove "data/" prefix
    ((processed++))
    
    echo "[$processed/${#images[@]}] $img_name" | tee -a "${LOG_FILE}"
    
    start=$(date +%s)
    
    if docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest \
        /runner/run.sh "/data/$rel_path" >> "${LOG_FILE}" 2>&1; then
        
        elapsed=$(($(date +%s) - start))
        
        if [ -f "data/out/designA/${img_name}.obj" ]; then
            mv "data/out/designA/${img_name}.obj" "${OUTPUT_DIR}/"
            echo "✓" | tee -a "${LOG_FILE}"
            echo "$elapsed  $img_name" >> "${TIME_LOG}"
            ((successful++))
        else
            echo "✗" | tee -a "${LOG_FILE}"
            ((failed++))
        fi
    else
        echo "✗" | tee -a "${LOG_FILE}"
        ((failed++))
    fi
done

batch_end=$(date +%s)
total=$((batch_end - batch_start))

echo "" | tee -a "${LOG_FILE}"
echo "Complete: $processed imgs, $successful OK, $failed failed" | tee -a "${LOG_FILE}"
[ $processed -gt 0 ] && echo "Rate: $((successful*100/processed))%" | tee -a "${LOG_FILE}"
echo "Time: $((total/60))m $((total%60))s" | tee -a "${LOG_FILE}"


batch_end_time=$(date +%s)
batch_elapsed=$((batch_end_time - batch_start_time))

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Batch Complete" | tee -a "${LOG_FILE}"
echo "Processed: $processed | Success: $successful | Failed: $failed" | tee -a "${LOG_FILE}"
if [ $processed -gt 0 ]; then
    SUCCESS_RATE=$((successful * 100 / processed))
    echo "Success rate: ${SUCCESS_RATE}%" | tee -a "${LOG_FILE}"
fi
echo "Time: $((batch_elapsed / 60))m $((batch_elapsed % 60))s" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "" >> "${TIME_LOG}"
echo "TOTAL_BATCH_TIME: ${batch_elapsed} sec" >> "${TIME_LOG}"
if [ $processed -gt 0 ]; then
    echo "SUCCESS_RATE: $((successful * 100 / processed))%" >> "${TIME_LOG}"
fi

echo "✅ Done! Results in ${OUTPUT_DIR}"
