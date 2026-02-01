#!/bin/bash
# Design A - Batch Processing Script for AFLW2000-3D Subset
# Processes images through VRN Docker container and logs timing
# Usage: ./batch_process_aflw2000.sh [subset_file]
# If no subset_file provided, processes all images in INPUT_DIR

set -e

# Configuration
INPUT_DIR="data/in/aflw2000"
OUTPUT_DIR="data/out/designA"
LOG_FILE="${OUTPUT_DIR}/batch_process.log"
TIME_LOG="${OUTPUT_DIR}/time.log"
SUBSET_FILE="${1:-}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Initialize logs
echo "Design A - Batch Processing Started: $(date)" | tee "${LOG_FILE}"
if [ -n "$SUBSET_FILE" ]; then
    echo "Using subset file: $SUBSET_FILE" | tee -a "${LOG_FILE}"
    SUBSET_SIZE=$(wc -l < "$SUBSET_FILE")
    echo "Processing $SUBSET_SIZE images from subset" | tee -a "${LOG_FILE}"
fi
echo "# Timing Log - Format: seconds KB filename" > "${TIME_LOG}"

# Batch timer
batch_start_time=$(date +%s)

# Counter for success/failure tracking
processed=0
successful=0
failed=0

# Collect input images
if [ -n "$SUBSET_FILE" ]; then
    # Use subset file
    mapfile -t images < "$SUBSET_FILE"
    # Prepend INPUT_DIR to each filename
    for i in "${!images[@]}"; do
        images[$i]="${INPUT_DIR}/${images[$i]}"
    done
else
    # Use all images in INPUT_DIR (avoid false negatives during loop)
    shopt -s nullglob
    images=("${INPUT_DIR}"/*.jpg)
fi

if [ ${#images[@]} -eq 0 ]; then
    echo "No images found in ${INPUT_DIR}" | tee -a "${LOG_FILE}"
    exit 1
fi

# Process each image
for img_path in "${images[@]}"; do
    
    img_name=$(basename "$img_path")
    echo "" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Processing: ${img_name}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
    
    # Run VRN with timing
    start_time=$(date +%s)
    if /usr/bin/time -f "%e sec  %M KB  ${img_name}" \
        docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest \
        /runner/run.sh "/data/in/aflw2000/${img_name}" 2>&1 | tee -a "${LOG_FILE}"; then
        
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Completed in ${elapsed} seconds" | tee -a "${LOG_FILE}"
        echo "${elapsed} sec  ${img_name}" >> "${TIME_LOG}"
        
        # Check if output was created
        if [ -f "${INPUT_DIR}/${img_name}.obj" ]; then
            mv "${INPUT_DIR}/${img_name}.obj" "${OUTPUT_DIR}/" 2>/dev/null || true
            successful=$((successful + 1))
            echo "✓ SUCCESS: ${img_name}.obj generated" | tee -a "${LOG_FILE}"
        else
            failed=$((failed + 1))
            echo "✗ FAILED: No .obj output for ${img_name}" | tee -a "${LOG_FILE}"
        fi
        
        # Move crop if exists
        if [ -f "${INPUT_DIR}/${img_name}.crop.jpg" ]; then
            mv "${INPUT_DIR}/${img_name}.crop.jpg" "${OUTPUT_DIR}/" 2>/dev/null || true
        fi
    else
        failed=$((failed + 1))
        echo "✗ FAILED: Docker execution error for ${img_name}" | tee -a "${LOG_FILE}"
    fi
    
    processed=$((processed + 1))
done

# Total batch time
batch_end_time=$(date +%s)
batch_elapsed=$((batch_end_time - batch_start_time))
echo "${batch_elapsed} sec  TOTAL_BATCH" >> "${TIME_LOG}"

# Summary
echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Batch Processing Complete: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Total processed: ${processed}" | tee -a "${LOG_FILE}"
echo "Successful: ${successful}" | tee -a "${LOG_FILE}"
echo "Failed: ${failed}" | tee -a "${LOG_FILE}"
echo "Success rate: $(awk "BEGIN {printf \"%.2f\", (${successful}/${processed})*100}")%" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Outputs saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Logs saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Timing data saved to: ${TIME_LOG}" | tee -a "${LOG_FILE}"
echo "Total batch time: ${batch_elapsed} seconds" | tee -a "${LOG_FILE}"

# Run metrics summary (includes Chamfer, F1_tau, F1_2tau)
echo "" | tee -a "${LOG_FILE}"
echo "================================================" | tee -a "${LOG_FILE}"
echo "Running Design A Metrics" | tee -a "${LOG_FILE}"
echo "================================================" | tee -a "${LOG_FILE}"
./scripts/analyze_results.sh | tee -a "${LOG_FILE}"
