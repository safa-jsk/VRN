#!/bin/bash
# Design C - Generate Teacher Volumes
# Run legacy VRN Docker to create reference volumes for distillation

set -e

# Configuration
INPUT_DIR="${1:-data/FaceScape/images}"
OUTPUT_DIR="${2:-data/FaceScape/teacher_volumes}"
CONTAINER_NAME="vrn-legacy"

echo "=== Design C: Generate Teacher Volumes ==="
echo "Input images: $INPUT_DIR"
echo "Output volumes: $OUTPUT_DIR"
echo

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo "Waiting for FaceScape images to finish downloading..."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Docker container exists (from Design A/B)
if ! docker images | grep -q "$CONTAINER_NAME"; then
    echo "Error: VRN Docker image '$CONTAINER_NAME' not found"
    echo "Please build Design A/B Docker container first"
    exit 1
fi

echo "Processing images with legacy VRN..."
echo "This will generate .raw volumes for teacher supervision"
echo

# Run VRN Docker (similar to Design B stage 1)
# Modify this based on actual VRN Docker interface
docker run --rm \
    -v "$(pwd)/$INPUT_DIR:/input" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    $CONTAINER_NAME \
    /path/to/vrn_process.sh /input /output

echo
echo "✓ Teacher volume generation complete"
echo "Converting .raw to .npy..."

# Convert .raw to .npy (reuse Design B volume_io)
python3 -c "
import numpy as np
from pathlib import Path
import sys
sys.path.append('designB/python')
from volume_io import load_raw_volume, save_volume_npy

output_dir = Path('$OUTPUT_DIR')
for raw_file in sorted(output_dir.glob('*.raw')):
    vol = load_raw_volume(raw_file)
    npy_file = raw_file.with_suffix('.npy')
    save_volume_npy(vol, npy_file)
    print(f'Converted: {raw_file.name} → {npy_file.name}')
"

echo
echo "✓ All teacher volumes ready for training"
echo "Location: $OUTPUT_DIR/*.npy"
