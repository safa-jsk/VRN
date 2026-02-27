#!/usr/bin/env python3
"""
[DESIGN.A][DESIGN.B][CAMFM.A3_METRICS] Mesh Quality Metrics
Computes Chamfer Distance, F1_tau, and F1_2tau between Design A meshes
and reference meshes (default: Design B).

Metrics computed:
- Chamfer Distance (CD): Geometric similarity measure
- F1_tau: F1 score at threshold tau (precision + recall)
- F1_2tau: F1 score at threshold 2*tau
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

try:
    import torch
except Exception as e:
    torch = None


def _load_chamfer_extension(chamfer_root: Path):
    """Attempt to load chamfer CUDA extension from chamfer/build."""
    if torch is None:
        return None

    build_dir = chamfer_root / "build"
    if not build_dir.exists():
        return None

    for file in build_dir.iterdir():
        if file.is_dir() and file.name.startswith("lib"):
            sys.path.insert(0, str(file))
            break

    try:
        import chamfer  # type: ignore
        return chamfer
    except Exception:
        return None


def _sample_points(mesh: trimesh.Trimesh, n_samples: int) -> np.ndarray:
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32)
    n_samples = max(1, min(n_samples, len(mesh.vertices)))
    return mesh.sample(n_samples).astype(np.float32)


def _chamfer_gpu(pred_pts: np.ndarray, ref_pts: np.ndarray, chamfer_mod, device: str):
    """Compute chamfer distances using CUDA extension (squared distances)."""
    if pred_pts.size == 0 or ref_pts.size == 0:
        return None

    pred = torch.from_numpy(pred_pts).unsqueeze(0).to(device)
    ref = torch.from_numpy(ref_pts).unsqueeze(0).to(device)

    dist1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.float32)
    dist2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.float32)
    idx1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.int32)
    idx2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.int32)

    chamfer_mod.forward(pred, ref, dist1, dist2, idx1, idx2)

    dist1 = dist1.squeeze(0)
    dist2 = dist2.squeeze(0)

    return dist1, dist2


def _chamfer_cpu(pred_pts: np.ndarray, ref_pts: np.ndarray):
    """Compute chamfer distances using CPU (Euclidean distances)."""
    if pred_pts.size == 0 or ref_pts.size == 0:
        return None

    from scipy.spatial import cKDTree

    tree_ref = cKDTree(ref_pts)
    tree_pred = cKDTree(pred_pts)

    dist1, _ = tree_ref.query(pred_pts, k=1)
    dist2, _ = tree_pred.query(ref_pts, k=1)

    return dist1, dist2


def _f1_scores(dist1, dist2, tau: float, squared: bool):
    if squared:
        thresh = tau * tau
        p = np.mean(dist1 <= thresh) if len(dist1) else 0.0
        r = np.mean(dist2 <= thresh) if len(dist2) else 0.0
    else:
        p = np.mean(dist1 <= tau) if len(dist1) else 0.0
        r = np.mean(dist2 <= tau) if len(dist2) else 0.0

    denom = p + r
    f1 = (2 * p * r / denom) if denom > 0 else 0.0
    return f1, p, r


def compute_metrics(pred_mesh_path: Path, ref_mesh_path: Path, n_samples: int, tau: float,
                    chamfer_mod=None, device: str = "cuda"):
    pred_mesh = trimesh.load(pred_mesh_path, force='mesh')
    ref_mesh = trimesh.load(ref_mesh_path, force='mesh')

    pred_pts = _sample_points(pred_mesh, n_samples)
    ref_pts = _sample_points(ref_mesh, n_samples)

    if pred_pts.size == 0 or ref_pts.size == 0:
        return {
            "status": "empty_mesh",
            "error": "Empty mesh or no sample points",
        }

    if chamfer_mod is not None and torch is not None and torch.cuda.is_available():
        dist1, dist2 = _chamfer_gpu(pred_pts, ref_pts, chamfer_mod, device)
        dist1_np = dist1.detach().cpu().numpy()
        dist2_np = dist2.detach().cpu().numpy()
        chamfer_mean_sq = float(0.5 * (dist1_np.mean() + dist2_np.mean()))
        chamfer_mean = float(0.5 * (np.sqrt(dist1_np).mean() + np.sqrt(dist2_np).mean()))
        f1_tau, p_tau, r_tau = _f1_scores(dist1_np, dist2_np, tau, squared=True)
        f1_2tau, p_2tau, r_2tau = _f1_scores(dist1_np, dist2_np, 2 * tau, squared=True)
        mode = "cuda"
    else:
        dist1, dist2 = _chamfer_cpu(pred_pts, ref_pts)
        chamfer_mean = float(0.5 * (dist1.mean() + dist2.mean()))
        chamfer_mean_sq = float(0.5 * ((dist1 ** 2).mean() + (dist2 ** 2).mean()))
        f1_tau, p_tau, r_tau = _f1_scores(dist1, dist2, tau, squared=False)
        f1_2tau, p_2tau, r_2tau = _f1_scores(dist1, dist2, 2 * tau, squared=False)
        mode = "cpu"

    return {
        "status": "ok",
        "mode": mode,
        "pred_vertices": len(pred_mesh.vertices),
        "ref_vertices": len(ref_mesh.vertices),
        "pred_faces": len(pred_mesh.faces),
        "ref_faces": len(ref_mesh.faces),
        "n_pred_samples": len(pred_pts),
        "n_ref_samples": len(ref_pts),
        "chamfer_mean": chamfer_mean,
        "chamfer_mean_sq": chamfer_mean_sq,
        "f1_tau": f1_tau,
        "f1_2tau": f1_2tau,
        "precision_tau": p_tau,
        "recall_tau": r_tau,
        "precision_2tau": p_2tau,
        "recall_2tau": r_2tau,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Design A mesh metrics (Chamfer, F1_tau, F1_2tau)"
    )
    parser.add_argument("--pred-dir", default="data/out/designA",
                        help="Directory with Design A .obj meshes")
    parser.add_argument("--ref-dir", default="data/out/designB/meshes",
                        help="Directory with reference .obj meshes (Design B)")
    parser.add_argument("--output-csv", default="data/out/designA/metrics/mesh_metrics.csv",
                        help="Output CSV for per-mesh metrics")
    parser.add_argument("--pattern", default="*.obj",
                        help="Glob pattern for mesh files")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of points to sample per mesh")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Distance threshold for F1_tau (same units as vertices)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU distance computation")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    chamfer_mod = None
    if not args.cpu_only:
        chamfer_root = Path(__file__).resolve().parents[1] / "chamfer"
        chamfer_mod = _load_chamfer_extension(chamfer_root)

    pred_meshes = sorted(pred_dir.glob(args.pattern))

    if not pred_meshes:
        print(f"No meshes found in {pred_dir} (pattern: {args.pattern})")

    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        print("Metrics will be marked as missing reference.")

    rows = []
    ok_rows = []
    missing = 0

    for pred_mesh in pred_meshes:
        ref_mesh = ref_dir / pred_mesh.name
        if not ref_mesh.exists():
            missing += 1
            rows.append({
                "name": pred_mesh.stem,
                "pred_path": str(pred_mesh),
                "ref_path": str(ref_mesh),
                "status": "missing_reference",
                "error": "Reference mesh not found",
            })
            continue

        try:
            metrics = compute_metrics(
                pred_mesh, ref_mesh,
                n_samples=args.samples,
                tau=args.tau,
                chamfer_mod=chamfer_mod,
                device="cuda"
            )
            metrics.update({
                "name": pred_mesh.stem,
                "pred_path": str(pred_mesh),
                "ref_path": str(ref_mesh),
                "tau": args.tau,
            })
            rows.append(metrics)
            if metrics.get("status") == "ok":
                ok_rows.append(metrics)
        except Exception as e:
            rows.append({
                "name": pred_mesh.stem,
                "pred_path": str(pred_mesh),
                "ref_path": str(ref_mesh),
                "status": "error",
                "error": str(e),
            })

    fieldnames = [
        "name", "pred_path", "ref_path", "status", "mode",
        "pred_vertices", "ref_vertices", "pred_faces", "ref_faces",
        "n_pred_samples", "n_ref_samples",
        "chamfer_mean", "chamfer_mean_sq",
        "f1_tau", "f1_2tau",
        "precision_tau", "recall_tau", "precision_2tau", "recall_2tau",
        "tau", "error"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n================================================")
    print("Design A Mesh Metrics Summary")
    print("================================================")
    print(f"Pred dir: {pred_dir}")
    print(f"Ref dir:  {ref_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Total meshes: {len(pred_meshes)}")
    print(f"Missing reference: {missing}")

    if ok_rows:
        chamfer_mean = np.mean([r["chamfer_mean"] for r in ok_rows])
        chamfer_mean_sq = np.mean([r["chamfer_mean_sq"] for r in ok_rows])
        f1_tau = np.mean([r["f1_tau"] for r in ok_rows])
        f1_2tau = np.mean([r["f1_2tau"] for r in ok_rows])
        mode = ok_rows[0].get("mode", "cpu")

        print(f"Compute mode: {mode}")
        print(f"Tau: {args.tau}")
        print(f"Chamfer (mean L2): {chamfer_mean:.6f}")
        print(f"Chamfer (mean squared): {chamfer_mean_sq:.6f}")
        print(f"F1_tau: {f1_tau:.6f}")
        print(f"F1_2tau: {f1_2tau:.6f}")
    else:
        print("No valid mesh pairs found. Check reference directory.")


if __name__ == "__main__":
    main()
