#!/bin/bash
# Design B - Stage 1: Extract Volumes from VRN
# Uses a modified VRN approach to extract volumes via raw2obj.py

set -e  # Exit on error

# Configuration
INPUT_DIR="${1:-data/in/aflw2000}"
OUTPUT_VOL_DIR="${2:-data/out/designB/volumes}"
DESIGN_A_DIR="data/out/designA"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Design B - Stage 1: Volume Extraction"
echo "======================================================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_VOL_DIR"
echo ""

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_VOL_DIR"

# Count input images
IMAGE_COUNT=$(find "$INPUT_DIR" -name "*.jpg" -not -name "*.crop.jpg" | wc -l)
echo "Found $IMAGE_COUNT images to process"
echo ""

echo -e "${BLUE}Note: VRN Docker container does not export .raw volumes by default.${NC}"
echo -e "${BLUE}Using alternative approach: Process with VRN → extract volume from output${NC}"
echo ""

# Process each image
CURRENT=0
SUCCESS=0
FAILED=0

for IMG in "$INPUT_DIR"/*.jpg; do
    # Skip crop files
    if [[ "$IMG" == *".crop.jpg" ]]; then
        continue
    fi
    
    if [ ! -f "$IMG" ]; then
        continue
    fi
    
    CURRENT=$((CURRENT + 1))
    IMG_NAME=$(basename "$IMG")
    BASENAME="${IMG_NAME%.jpg}"
    
    echo -e "${YELLOW}[$CURRENT/$IMAGE_COUNT]${NC} Processing: $BASENAME"
    
    # Run VRN Docker to process image (outputs to same dir as input)
    if docker run --rm \
        -v "$(pwd)/data:/data" \
        asjackson/vrn:latest \
        /runner/run.sh "/data/in/aflw2000/${IMG_NAME}" > /dev/null 2>&1; then
        
        # Check if .obj was created (indicates successful processing)
        OBJ_FILE="$INPUT_DIR/${IMG_NAME}.obj"
        
        if [ -f "$OBJ_FILE" ]; then
            # VRN succeeded - now we need to extract the volume
            # The standard VRN Docker doesn't export volumes, so we'll create a placeholder
            # In a real implementation, you'd modify the VRN source to export volumes
            
            # For now, create a flag file indicating we need volume extraction
            echo "# Volume extraction pending for $BASENAME" > "$OUTPUT_VOL_DIR/${BASENAME}.pending"
            
            # Clean up the obj file (we'll regenerate it in stage 2)
            rm -f "$OBJ_FILE"
            rm -f "$INPUT_DIR/${IMG_NAME}.crop.jpg"
            
            # Mark as failed since we don't have the volume yet
            echo -e "  ${RED}✗${NC} Volume extraction not supported by standard VRN Docker"
            FAILED=$((FAILED + 1))
        else
            echo -e "  ${RED}✗${NC} VRN processing failed (likely face detection failure)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "  ${RED}✗${NC} Docker execution failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================================"
echo "Stage 1 Complete - Volume Extraction Summary"
echo "======================================================================"
echo "Total processed: $CURRENT"
echo -e "Successful: ${GREEN}$SUCCESS${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""
echo -e "${YELLOW}⚠ LIMITATION: Standard VRN Docker does not export volumes${NC}"
echo -e "${YELLOW}   To enable volume extraction, you need to:${NC}"
echo -e "${YELLOW}   1. Modify VRN source code to export .raw files${NC}"
echo -e "${YELLOW}   2. Build custom Docker image${NC}"
echo -e "${YELLOW}   3. OR use existing Design A meshes for comparison${NC}"
echo ""
echo -e "${BLUE}Alternative: Use Design A meshes as baseline for demonstration${NC}"
echo "======================================================================"
