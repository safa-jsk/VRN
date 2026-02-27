#!/usr/bin/env bash
# Smoke test – verify imports and basic pipeline logic
# Run from repo root: bash scripts/smoke_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0; FAIL=0
ok()   { echo "  ✓ $1"; ((PASS++)); }
fail() { echo "  ✗ $1"; ((FAIL++)); }

echo "VRN Smoke Tests"
echo "==============="

# ── 1. Core imports ──
echo ""
echo "1. Core imports:"
python3 -c "from src.vrn.config import REPO_ROOT; print(f'  config → {REPO_ROOT}')" && ok "src.vrn.config" || fail "src.vrn.config"
python3 -c "from src.vrn.io import load_volume_npy" && ok "src.vrn.io" || fail "src.vrn.io"
python3 -c "from src.vrn.perf import configure_gpu, warmup, timed_call" && ok "src.vrn.perf" || fail "src.vrn.perf"
python3 -c "from src.vrn.metrics import f1_score" && ok "src.vrn.metrics" || fail "src.vrn.metrics"
python3 -c "from src.vrn.utils import check_cuda" && ok "src.vrn.utils" || fail "src.vrn.utils"

# ── 2. DesignB imports ──
echo ""
echo "2. DesignB imports:"
python3 -c "from src.designB.io import load_volume_npy, save_mesh_obj" && ok "src.designB.io" || fail "src.designB.io"
python3 -c "from src.designB.pipeline import marching_cubes_baseline" && ok "src.designB.pipeline (baseline)" || fail "src.designB.pipeline"

# ── 3. DesignC skeleton ──
echo ""
echo "3. DesignC skeleton:"
python3 -c "from src.designC.infer_facescape import main" && ok "src.designC.infer_facescape" || fail "src.designC.infer_facescape"
python3 -c "from src.designC.eval_facescape import main" && ok "src.designC.eval_facescape" || fail "src.designC.eval_facescape"

# ── 4. Config defaults ──
echo ""
echo "4. Config defaults:"
python3 -c "
from src.vrn.config import load_config
c = load_config()
assert c['threshold'] == 0.5, f'Expected 0.5, got {c[\"threshold\"]}'
assert c['warmup_iters'] == 15, f'Expected 15, got {c[\"warmup_iters\"]}'
print('  threshold=0.5, warmup_iters=15')
" && ok "defaults correct" || fail "defaults"

# ── 5. Pipeline --help ──
echo ""
echo "5. CLI --help:"
python3 -m src.designB.pipeline --help > /dev/null 2>&1 && ok "pipeline --help" || fail "pipeline --help"
python3 -m src.designB.benchmark --help > /dev/null 2>&1 && ok "benchmark --help" || fail "benchmark --help"

# ── Summary ──
echo ""
echo "==============="
echo "Passed: $PASS  Failed: $FAIL"
if [[ $FAIL -gt 0 ]]; then
    echo "⚠ Some smoke tests failed."
    exit 1
else
    echo "✓ All smoke tests passed."
fi
