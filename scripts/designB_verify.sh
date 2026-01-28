#!/bin/bash
# Design B - Verification Script
# Runs mesh verification and generates comparison reports

set -e

DESIGN_A_DIR="${1:-data/out/designA}"
DESIGN_B_DIR="${2:-data/out/designB/meshes}"
OUTPUT_DIR="${3:-data/out/designB}"

echo "======================================================================"
echo "Design B - Mesh Verification"
echo "======================================================================"
echo "Design A (baseline): $DESIGN_A_DIR"
echo "Design B (CUDA):     $DESIGN_B_DIR"
echo ""

# Check directories exist
if [ ! -d "$DESIGN_A_DIR" ]; then
    echo "Error: Design A directory not found: $DESIGN_A_DIR"
    exit 1
fi

if [ ! -d "$DESIGN_B_DIR" ]; then
    echo "Error: Design B directory not found: $DESIGN_B_DIR"
    exit 1
fi

# Count meshes
A_COUNT=$(ls -1 "$DESIGN_A_DIR"/*.obj 2>/dev/null | wc -l)
B_COUNT=$(ls -1 "$DESIGN_B_DIR"/*.obj 2>/dev/null | wc -l)

echo "Mesh counts:"
echo "  Design A: $A_COUNT"
echo "  Design B: $B_COUNT"
echo ""

# Run verification
echo "Running verification..."
python3 cuda_post/verify_meshes.py \
    --designA "$DESIGN_A_DIR" \
    --designB "$DESIGN_B_DIR" \
    --output "$OUTPUT_DIR/verification.json"

echo ""
echo "======================================================================"
echo "Verification complete"
echo "======================================================================"
echo "Results saved to: $OUTPUT_DIR/verification.json"
