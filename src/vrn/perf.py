"""
CAMFM performance controls: warmup, CUDA sync timing, optional AMP/compile.

[CAMFM.A2b_STEADY_STATE]
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import torch


def configure_gpu(
    cudnn_benchmark: bool = True,
    tf32: bool = True,
) -> None:
    """Apply GPU-level performance flags (CAMFM.A2b)."""
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def warmup(
    fn: Callable,
    *args,
    iters: int = 15,
    verbose: bool = True,
    **kwargs,
) -> float:
    """Run *fn* for *iters* iterations to reach GPU steady-state.

    Returns wall-clock warmup time (excluded from benchmarks).
    """
    if iters <= 0:
        return 0.0
    if verbose:
        print(f"[CAMFM.A2b] Warmup: {iters} iterations …")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    if verbose:
        print(f"[CAMFM.A2b] Warmup done: {elapsed:.3f}s")
    return elapsed


def timed_call(
    fn: Callable,
    *args,
    **kwargs,
) -> Tuple[float, object]:
    """Execute *fn* with proper CUDA synchronisation and return (elapsed_s, result)."""
    torch.cuda.synchronize()
    t0 = time.time()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed, result
