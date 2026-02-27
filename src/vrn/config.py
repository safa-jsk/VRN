"""
Central configuration and path handling for the VRN project.

Every script/module should resolve paths through this module to avoid
hardcoded absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # pip install pyyaml (listed in requirements.txt)

# ---------------------------------------------------------------------------
# Repo-root resolution
# ---------------------------------------------------------------------------

def get_repo_root() -> Path:
    """Return the repository root directory.

    Resolution order:
    1. ``VRN_ROOT`` environment variable (if set).
    2. Walk upward from this file until we find ``.git/``.
    """
    env = os.environ.get("VRN_ROOT")
    if env:
        return Path(env).resolve()

    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(
        "Cannot locate repository root. Set VRN_ROOT or run from inside the repo."
    )


REPO_ROOT = get_repo_root()


# ---------------------------------------------------------------------------
# Standard paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
BUILD_DIR = ARTIFACTS_DIR / "build"
TEMP_DIR = ARTIFACTS_DIR / "temp"

ASSETS_DIR = REPO_ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"
DATA_DIR = ASSETS_DIR / "data"

EXTERNAL_DIR = REPO_ROOT / "external"
MC_EXT_DIR = EXTERNAL_DIR / "marching_cubes_cuda_ext"
CHAMFER_EXT_DIR = EXTERNAL_DIR / "chamfer_ext"

DESIGNB_CONFIGS_DIR = REPO_ROOT / "DesignB" / "configs"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "threshold": 0.5,
    "volume_shape": [200, 192, 192],
    "warmup_iters": 15,
    "benchmark_runs": 3,
    "cudnn_benchmark": True,
    "tf32": True,
    "amp": False,
    "compile": False,
    "device": "cuda",
}


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load a YAML config, falling back to built-in defaults.

    Parameters
    ----------
    path : Path, optional
        Explicit config file.  When *None* the default
        ``DesignB/configs/default.yaml`` is used.
    """
    if path is None:
        path = DESIGNB_CONFIGS_DIR / "default.yaml"
    cfg = dict(_DEFAULT_CONFIG)
    if path.exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg
