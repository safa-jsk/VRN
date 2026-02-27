#!/bin/bash
# Design B - Batch Processing for 300W_LP AFW Dataset (1000 samples)
# Two-stage pipeline: VRN Volume Extraction → CUDA Marching Cubes
#
# Usage: ./scripts/designB_batch_300w_afw.sh [paths_file]
#        Default: docs/300w_afw_1000_paths.txt

# Don't use set -e here to handle docker failures gracefully

# Configuration
PATHS_FILE="${1:-docs/300w_afw_1000_paths.txt}"
OUTPUT_BASE="data/out/designB_300w_afw"
VOLUME_DIR="${OUTPUT_BASE}/volumes"
MESH_DIR="${OUTPUT_BASE}/meshes"
LOG_DIR="${OUTPUT_BASE}/logs"
TIMING_LOG="${LOG_DIR}/timing.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "======================================================================"
echo -e "${BLUE}Design B - 300W_LP AFW Batch Processing${NC}"
echo "======================================================================"
echo "Paths file: $PATHS_FILE"
echo "Output: $OUTPUT_BASE"
echo ""

# Validate paths file
if [ ! -f "$PATHS_FILE" ]; then
    echo -e "${RED}ERROR: Paths file not found: $PATHS_FILE${NC}"
    exit 1
fi

# Count images
TOTAL=$(wc -l < "$PATHS_FILE")
echo "Total images: $TOTAL"
echo ""

# Create output directories
mkdir -p "$VOLUME_DIR" "$MESH_DIR" "$LOG_DIR"

# Initialize logs
echo "# Design B - 300W_LP AFW Batch Log" > "${LOG_DIR}/batch.log"
echo "# Started: $(date)" >> "${LOG_DIR}/batch.log"
echo "# Paths file: $PATHS_FILE" >> "${LOG_DIR}/batch.log"
echo "# Total images: $TOTAL" >> "${LOG_DIR}/batch.log"
echo "" >> "${LOG_DIR}/batch.log"

echo "# Timing Log - Format: stage elapsed_sec filename" > "$TIMING_LOG"

# ======================================================================
# STAGE 1: VRN Volume Extraction
# ======================================================================
echo "======================================================================"
echo -e "${BLUE}STAGE 1: VRN Volume Extraction${NC}"
echo "======================================================================"

STAGE1_START=$(date +%s)
STAGE1_SUCCESS=0
STAGE1_FAIL=0
count=0

# Read all paths into array first (avoids stdin issues with Docker)
mapfile -t IMAGES < "$PATHS_FILE"
echo "Loaded ${#IMAGES[@]} image paths"
echo ""

# Process each image
for img_path in "${IMAGES[@]}"; do
    [ -z "$img_path" ] && continue
    
    img_name=$(basename "$img_path")
    img_name_noext="${img_name%.*}"
    rel_path=${img_path#data/}
    ((count++))
    
    # Skip if mesh already exists
    if [ -f "${MESH_DIR}/${img_name}.obj" ]; then
        echo "[$count/$TOTAL] $img_name (cached)"
        ((STAGE1_SUCCESS++))
        continue
    fi
    
    echo -n "[$count/$TOTAL] $img_name ... "
    
    start_time=$(date +%s.%N)
    
    # Run VRN Docker to process image
    # Using </dev/null to prevent Docker from consuming stdin
    if docker run --rm \
        -v "$PWD/data:/data" \
        asjackson/vrn:latest \
        /runner/run.sh "/data/$rel_path" </dev/null >> "${LOG_DIR}/vrn_stage1.log" 2>&1; then
        
        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)
        
        # VRN outputs .obj files next to input or in designA folder
        # Check if .obj was created and move it
        if [ -f "${img_path}.obj" ]; then
            # Move mesh
            mv "${img_path}.obj" "${MESH_DIR}/${img_name}.obj" 2>/dev/null || true
            echo -e "${GREEN}✓${NC} ${elapsed}s"
            echo "stage1 $elapsed $img_name" >> "$TIMING_LOG"
            ((STAGE1_SUCCESS++))
        elif [ -f "data/out/designA/${img_name}.obj" ]; then
            mv "data/out/designA/${img_name}.obj" "${MESH_DIR}/${img_name}.obj" 2>/dev/null || true
            echo -e "${GREEN}✓${NC} ${elapsed}s"
            echo "stage1 $elapsed $img_name" >> "$TIMING_LOG"
            ((STAGE1_SUCCESS++))
        else
            echo -e "${RED}✗${NC} (no output)"
            echo "FAIL: $img_name - no .obj generated" >> "${LOG_DIR}/batch.log"
            ((STAGE1_FAIL++))
        fi
    else
        echo -e "${RED}✗${NC}"
        echo "FAIL: $img_name - docker error" >> "${LOG_DIR}/batch.log"
        ((STAGE1_FAIL++))
    fi
    
done

STAGE1_END=$(date +%s)
STAGE1_TIME=$((STAGE1_END - STAGE1_START))

echo ""
echo "Stage 1 Complete:"
echo "  Success: $STAGE1_SUCCESS"
echo "  Failed: $STAGE1_FAIL"
echo "  Time: $((STAGE1_TIME/60))m $((STAGE1_TIME%60))s"
echo ""

# ======================================================================
# STAGE 2: GPU Marching Cubes (if we have volumes)
# ======================================================================
# Note: Standard VRN Docker doesn't export volumes, only meshes.
# For Design B demonstration, we use the meshes from Stage 1.
# 
# If you have modified VRN to export volumes (.raw or .npy), 
# uncomment the following section.

echo "======================================================================"
echo -e "${BLUE}STAGE 2: Post-Processing Summary${NC}"
echo "======================================================================"

# Count meshes created
MESH_COUNT=$(find "$MESH_DIR" -name "*.obj" 2>/dev/null | wc -l)

echo "Meshes generated: $MESH_COUNT"
echo ""

# If volumes exist, run CUDA marching cubes
VOLUME_COUNT=$(find "$VOLUME_DIR" -name "*.npy" 2>/dev/null | wc -l)

if [ "$VOLUME_COUNT" -gt 0 ]; then
    echo "Found $VOLUME_COUNT volumes, running CUDA marching cubes..."
    
    # Activate Python environment
    source vrn_env/bin/activate
    
    STAGE2_START=$(date +%s)
    
    # Run GPU marching cubes
    python3 designB/python/marching_cubes_cuda.py \
        --input "$VOLUME_DIR" \
        --output "${MESH_DIR}_cuda" \
        --threshold 0.5 \
        --image-dir "data/300W_LP/AFW" \
        2>&1 | tee "${LOG_DIR}/cuda_mc.log"
    
    STAGE2_END=$(date +%s)
    STAGE2_TIME=$((STAGE2_END - STAGE2_START))
    
    echo ""
    echo "Stage 2 Complete:"
    echo "  Time: $((STAGE2_TIME/60))m $((STAGE2_TIME%60))s"
else
    echo -e "${YELLOW}Note: No .npy volumes found in $VOLUME_DIR${NC}"
    echo "Standard VRN Docker outputs meshes directly."
    echo "For CUDA acceleration, modify VRN to export volumes."
    STAGE2_TIME=0
fi

# ======================================================================
# SUMMARY
# ======================================================================
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - STAGE1_START))

echo ""
echo "======================================================================"
echo -e "${BLUE}BATCH PROCESSING COMPLETE${NC}"
echo "======================================================================"
echo "Dataset: 300W_LP AFW ($TOTAL images)"
echo ""
echo "Stage 1 (VRN): $((STAGE1_TIME/60))m $((STAGE1_TIME%60))s"
echo "  Success: $STAGE1_SUCCESS / $TOTAL"
[ "$TOTAL" -gt 0 ] && echo "  Rate: $((STAGE1_SUCCESS*100/TOTAL))%"
echo ""
if [ "$STAGE2_TIME" -gt 0 ]; then
    echo "Stage 2 (CUDA MC): $((STAGE2_TIME/60))m $((STAGE2_TIME%60))s"
    echo ""
fi
echo "Total time: $((TOTAL_TIME/60))m $((TOTAL_TIME%60))s"
echo ""
echo "Output locations:"
echo "  Meshes: $MESH_DIR"
echo "  Logs: $LOG_DIR"
echo ""

# Save summary to log
echo "" >> "${LOG_DIR}/batch.log"
echo "# Completed: $(date)" >> "${LOG_DIR}/batch.log"
echo "# Success: $STAGE1_SUCCESS / $TOTAL" >> "${LOG_DIR}/batch.log"
echo "# Total time: ${TOTAL_TIME}s" >> "${LOG_DIR}/batch.log"
