"""Miscellaneous utilities."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_ext_importable() -> None:
    """Add the CUDA extension build directory to ``sys.path``
    so ``import marching_cubes_cuda_ext`` succeeds at runtime.
    """
    from src.vrn.config import BUILD_DIR, MC_EXT_DIR

    for candidate in [BUILD_DIR, MC_EXT_DIR]:
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)
    # Also check for in-tree .so from `pip install -e external/marching_cubes_cuda_ext`
    ext_build = MC_EXT_DIR / "build"
    if ext_build.exists():
        for child in ext_build.iterdir():
            if child.is_dir() and child.name.startswith("lib"):
                s = str(child)
                if s not in sys.path:
                    sys.path.insert(0, s)


def check_cuda(strict: bool = False) -> bool:
    """Return True if CUDA is available.  When *strict*, raise instead."""
    import torch

    ok = torch.cuda.is_available()
    if not ok and strict:
        raise RuntimeError("CUDA is required but not available.")
    return ok
