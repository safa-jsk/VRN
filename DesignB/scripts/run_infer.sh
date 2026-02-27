#!/usr/bin/env bash
# DesignB – single-volume inference
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT="${1:?Usage: run_infer.sh <input.npy> [output.obj] [threshold]}"
OUTPUT="${2:-artifacts/meshes/$(basename "${INPUT%.npy}").obj}"
THRESH="${3:-0.5}"

mkdir -p "$(dirname "$OUTPUT")"
python3 -m src.designB.pipeline \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --threshold "$THRESH"
