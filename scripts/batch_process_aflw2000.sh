#!/bin/bash
# Design A - Batch Processing Script for AFLW2000-3D Subset
# Processes images through VRN Docker container and logs timing

set -e

# Configuration
INPUT_DIR="data/in/aflw2000"
OUTPUT_DIR="data/out/designA"
LOG_FILE="${OUTPUT_DIR}/batch_process.log"
TIME_LOG="${OUTPUT_DIR}/time.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Initialize logs
echo "Design A - Batch Processing Started: $(date)" | tee "${LOG_FILE}"
echo "# Timing Log - Format: seconds KB filename" > "${TIME_LOG}"

# Counter for success/failure tracking
processed=0
successful=0
failed=0

# Process each image
for img_path in ${INPUT_DIR}/*.jpg; do
    if [ ! -f "$img_path" ]; then
        echo "No images found in ${INPUT_DIR}" | tee -a "${LOG_FILE}"
        exit 1
    fi
    
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
