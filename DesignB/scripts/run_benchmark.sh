#!/usr/bin/env bash
# DesignB – CPU vs GPU benchmark
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${1:-artifacts/volumes}"
OUTPUT_DIR="${2:-artifacts/benchmarks}"
WARMUP="${3:-15}"

mkdir -p "$OUTPUT_DIR"
python3 -m src.designB.benchmark \
    --input_dir  "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_iters "$WARMUP"
