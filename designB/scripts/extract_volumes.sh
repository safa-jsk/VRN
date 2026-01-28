#!/bin/bash
# Design B - Stage 1: Extract Volumes from VRN
# This is the CORRECT implementation using raw AFLW2000 images

set -e

echo "=========================================="
echo "Design B - Stage 1: VRN Volume Extraction"
echo "=========================================="

INPUT_DIR="data/in/aflw2000"
OUTPUT_DIR="data/out/designB/volumes_raw"
SUBSET_FILE="docs/aflw2000_subset.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count images
TOTAL=$(wc -l < "$SUBSET_FILE")
echo "Processing $TOTAL images from AFLW2000 subset..."
echo ""

# Run VRN to extract volumes (not meshes!)
# Override Docker entrypoint to run process.lua directly
docker run --rm \
    --entrypoint="" \
    -v "$(pwd)/$INPUT_DIR:/input" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    -v "$(pwd)/process.lua:/app/process.lua" \
    asjackson/vrn:latest \
    /root/usr/local/torch/install/bin/th /app/process.lua \
        --model /runner/vrn-unguided.t7 \
        --input /input \
        --output /output \
        --device cpu

echo ""
echo "=========================================="
echo "Volume Extraction Complete"
echo "=========================================="

# Count successful volumes
VOLUMES=$(find "$OUTPUT_DIR" -name "*.raw" | wc -l)
echo "Extracted: $VOLUMES/$TOTAL volumes"

# Convert .raw to .npy for easier Python processing
echo ""
echo "Converting .raw to .npy..."
python3 cuda_post/convert_raw_to_npy.py \
    --input "$OUTPUT_DIR" \
    --output "data/out/designB/volumes"

echo ""
echo "✓ Stage 1 Complete"
echo "  Volumes saved to: data/out/designB/volumes/"
