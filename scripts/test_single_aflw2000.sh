#!/bin/bash
# Quick test script - process just one AFLW2000 image to verify pipeline

set -e

INPUT_DIR="data/in/aflw2000"
OUTPUT_DIR="data/out/designA"

mkdir -p "${OUTPUT_DIR}"

# Get first image
test_img=$(ls ${INPUT_DIR}/*.jpg | head -1)
test_name=$(basename "$test_img")

echo "Testing VRN with: ${test_name}"
echo "================================"

# Run VRN
/usr/bin/time -v docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest \
    /runner/run.sh "/data/in/aflw2000/${test_name}"

echo ""
echo "Checking outputs..."

if [ -f "${INPUT_DIR}/${test_name}.obj" ]; then
    echo "✓ Mesh generated: ${test_name}.obj"
    ls -lh "${INPUT_DIR}/${test_name}.obj"
    mv "${INPUT_DIR}/${test_name}.obj" "${OUTPUT_DIR}/"
else
    echo "✗ No mesh output found"
fi

if [ -f "${INPUT_DIR}/${test_name}.crop.jpg" ]; then
    echo "✓ Crop generated: ${test_name}.crop.jpg"
    mv "${INPUT_DIR}/${test_name}.crop.jpg" "${OUTPUT_DIR}/"
fi

echo ""
echo "Test complete. Output in: ${OUTPUT_DIR}"
