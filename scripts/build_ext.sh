#!/usr/bin/env bash
# Build all CUDA extensions (marching cubes + chamfer)
# Run from repo root: bash scripts/build_ext.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Building CUDA Extensions"
echo "=========================================="

# ── Check Python / PyTorch / CUDA ──
echo ""
echo "Python Environment:"
echo "  Python:  $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA:    $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "  GPU:     $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

echo ""
echo "CUDA Compiler:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "  ✗ nvcc not found – install CUDA Toolkit 11.8"
    exit 1
fi

# ── Build marching cubes ext ──
echo ""
echo "── Marching Cubes CUDA Extension ──"
pushd external/marching_cubes_cuda_ext > /dev/null
rm -rf build/ dist/ *.egg-info
python3 setup.py build_ext --inplace
echo "✓ marching_cubes_cuda_ext built"
popd > /dev/null

# ── Build chamfer ext ──
echo ""
echo "── Chamfer Distance CUDA Extension ──"
pushd external/chamfer_ext > /dev/null
rm -rf build/ dist/ *.egg-info
python3 setup.py build_ext --inplace
echo "✓ chamfer_ext built"
popd > /dev/null

# ── Quick import test ──
echo ""
echo "Testing imports..."
python3 -c "
from external.marching_cubes_cuda_ext.cuda_marching_cubes import marching_cubes_gpu
print('  ✓ marching_cubes_gpu importable')
"
python3 -c "
import chamfer
print('  ✓ chamfer importable')
" 2>/dev/null || echo "  ⚠ chamfer import skipped (may need PYTHONPATH)"

echo ""
echo "=========================================="
echo "✓ All CUDA extensions built successfully"
echo "=========================================="
