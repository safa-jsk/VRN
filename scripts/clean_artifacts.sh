#!/usr/bin/env bash
# Clean generated artefacts (meshes, benchmarks, volumes, build outputs)
# Run from repo root: bash scripts/clean_artifacts.sh [--all]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Cleaning VRN artefacts..."

# Always clean
rm -rf artifacts/meshes/*
rm -rf artifacts/benchmarks/*
rm -rf artifacts/eval/*
echo "  ✓ Cleared artifacts/{meshes,benchmarks,eval}"

if [[ "${1:-}" == "--all" ]]; then
    # Also clean build outputs and volumes
    rm -rf artifacts/volumes/*
    rm -rf external/marching_cubes_cuda_ext/build/
    rm -rf external/marching_cubes_cuda_ext/dist/
    rm -rf external/marching_cubes_cuda_ext/*.egg-info
    rm -rf external/chamfer_ext/build/
    rm -rf external/chamfer_ext/dist/
    rm -rf external/chamfer_ext/*.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ Cleared volumes, build dirs, __pycache__"
fi

echo "✓ Clean complete"
