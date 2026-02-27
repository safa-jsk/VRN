"""
Shared input / output utilities.

Thin wrappers re-exported from the DesignB I/O module so that other
Design variants (and experiments) can import from a single place.
"""

from __future__ import annotations

from src.designB.io import (  # noqa: F401
    load_raw_volume,
    load_volume_npy,
    save_volume_npy,
    volume_to_tensor,
    save_mesh_obj,
    load_mesh_obj,
    get_mesh_stats,
)
