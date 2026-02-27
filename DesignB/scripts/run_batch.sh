#!/usr/bin/env bash
# DesignB – batch process all volumes
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${1:-artifacts/volumes}"
OUTPUT_DIR="${2:-artifacts/meshes}"
THRESH="${3:-0.5}"

mkdir -p "$OUTPUT_DIR"
python3 -m src.designB.pipeline \
    --batch \
    --input_dir  "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --threshold "$THRESH"
