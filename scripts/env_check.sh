#!/usr/bin/env bash
# Environment pre-flight check for VRN
# Run from repo root: bash scripts/env_check.sh
set -euo pipefail

PASS=0; FAIL=0
ok()   { echo "  ✓ $1"; ((PASS++)); }
fail() { echo "  ✗ $1"; ((FAIL++)); }

echo "VRN Environment Check"
echo "====================="

# ── OS ──
echo ""
echo "OS:"
if grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    ok "$(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"')"
else
    fail "Expected Ubuntu (found $(uname -s))"
fi

# ── Python ──
echo ""
echo "Python:"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    ok "$PY_VER"
else
    fail "python3 not found"
fi

# ── PyTorch ──
echo ""
echo "PyTorch:"
python3 -c "
import torch, sys
print(f'  ✓ torch {torch.__version__}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA {torch.version.cuda}')
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  ✓ Compute capability: sm_{cap[0]}{cap[1]}')
else:
    print('  ✗ CUDA not available')
    sys.exit(1)
" 2>/dev/null && ((PASS+=4)) || ((FAIL++))

# ── nvcc ──
echo ""
echo "CUDA Compiler:"
if command -v nvcc &>/dev/null; then
    ok "$(nvcc --version 2>&1 | grep 'release' | head -1 | xargs)"
else
    fail "nvcc not found (install CUDA Toolkit 11.8)"
fi

# ── Key Python packages ──
echo ""
echo "Python Packages:"
for pkg in numpy trimesh scikit-image Pillow matplotlib pyyaml; do
    if python3 -c "import importlib; importlib.import_module('${pkg//-/_}')" 2>/dev/null; then
        ok "$pkg"
    else
        fail "$pkg (pip install $pkg)"
    fi
done

# ── CUDA extension ──
echo ""
echo "CUDA Extensions:"
if python3 -c "from external.marching_cubes_cuda_ext.cuda_marching_cubes import marching_cubes_gpu" 2>/dev/null; then
    ok "marching_cubes_cuda_ext"
else
    fail "marching_cubes_cuda_ext (run: bash scripts/build_ext.sh)"
fi

# ── Summary ──
echo ""
echo "====================="
echo "Passed: $PASS  Failed: $FAIL"
if [[ $FAIL -gt 0 ]]; then
    echo "⚠ Fix the failures above before running the pipeline."
    exit 1
else
    echo "✓ Environment ready."
fi
