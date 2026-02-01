#!/bin/bash
# Design A - Simple Batch for 300W_LP (takes full path list)
# Usage: ./batch_process_300w_simple.sh <file_with_full_paths>

PATHS_FILE="${1:-}"
OUTPUT_DIR="data/out/designA_300w_lp"
LOG_FILE="${OUTPUT_DIR}/batch_process.log"
TIME_LOG="${OUTPUT_DIR}/time.log"

mkdir -p "${OUTPUT_DIR}"

echo "Design A - 300W_LP Batch: $(date)" | tee "${LOG_FILE}"

# Read full paths
mapfile -t images < "$PATHS_FILE"
echo "Loaded ${#images[@]} images" | tee -a "${LOG_FILE}"

if [ ${#images[@]} -eq 0 ]; then
    echo "ERROR: No images" | tee -a "${LOG_FILE}"
    exit 1
fi

batch_start=$(date +%s)
success=0
fail=0
count=0

for img_path in "${images[@]}"; do
    [ -z "$img_path" ] && continue
    img_name=$(basename "$img_path")
    rel_path=${img_path#data/}
    ((count++))
    
    echo "[$count/${#images[@]}] $img_name"
    
    start=$(date +%s)
    
    if docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest \
        /runner/run.sh "/data/$rel_path" >> "${LOG_FILE}" 2>&1; then
        
        # Check if output was created (Docker outputs next to source image)
        if [ -f "${img_path}.obj" ]; then
            mv "${img_path}.obj" "${OUTPUT_DIR}/"
            elapsed=$(($(date +%s) - start))
            echo "$elapsed  $img_name" >> "${TIME_LOG}"
            ((success++))
            echo "✓ $elapsed sec"
        elif [ -f "data/out/designA/${img_name}.obj" ]; then
            # Fallback: check default designA location
            mv "data/out/designA/${img_name}.obj" "${OUTPUT_DIR}/"
            elapsed=$(($(date +%s) - start))
            echo "$elapsed  $img_name" >> "${TIME_LOG}"
            ((success++))
            echo "✓ $elapsed sec"
        else
            ((fail++))
            echo "✗"
        fi
    else
        ((fail++))
        echo "✗"
    fi
done

total=$(($(date +%s) - batch_start))

echo "" | tee -a "${LOG_FILE}"
echo "Complete: $count images, $success OK, $fail failed" | tee -a "${LOG_FILE}"
[ $count -gt 0 ] && echo "Success rate: $((success*100/count))%" | tee -a "${LOG_FILE}"
echo "Total time: $((total/60))m $((total%60))s" | tee -a "${LOG_FILE}"
