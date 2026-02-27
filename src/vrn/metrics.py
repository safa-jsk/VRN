"""
Evaluation metric helpers (Chamfer Distance, F1 scores).

Lightweight wrappers; heavy computation stays in
``experiments/comparisons/designA_mesh_metrics.py``.
"""

from __future__ import annotations

import numpy as np


def f1_score(dist_pred_to_ref: np.ndarray, dist_ref_to_pred: np.ndarray, tau: float) -> float:
    """Compute F1 score at threshold *tau* (Euclidean distances)."""
    if dist_pred_to_ref.size == 0 or dist_ref_to_pred.size == 0:
        return 0.0
    precision = float(np.mean(dist_pred_to_ref <= tau))
    recall = float(np.mean(dist_ref_to_pred <= tau))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
