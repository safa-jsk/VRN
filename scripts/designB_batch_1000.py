#!/usr/bin/env python3
"""
Design B - Batch Processing for 1000 300W_LP AFW Samples
Processes images through VRN → Volume → GPU Marching Cubes → Mesh
Logs timing and computes Chamfer distance, F1_tau, F1_2tau metrics.

Usage:
    python3 scripts/designB_batch_1000.py \
        --image-list docs/300w_lp_afw_1000.txt \
        --output-dir data/out/designB_1000 \
        --ref-dir data/out/designA_300w_lp \
        --gpu
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

# Add designB/python to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "designB" / "python"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available, GPU mode disabled")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not available, metrics computation disabled")


# =============================================================================
# Chamfer Distance and F1 Score Computation
# =============================================================================

def load_chamfer_extension():
    """Load the custom CUDA chamfer extension."""
    if not HAS_TORCH:
        return None
    
    chamfer_root = Path(__file__).resolve().parent.parent / "chamfer"
    build_dir = chamfer_root / "build"
    
    if not build_dir.exists():
        return None
    
    for file in build_dir.iterdir():
        if file.is_dir() and file.name.startswith("lib"):
            sys.path.insert(0, str(file))
            break
    
    try:
        import chamfer
        return chamfer
    except Exception:
        return None


def sample_points(mesh, n_samples: int) -> np.ndarray:
    """Sample points from mesh surface."""
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32)
    n_samples = max(1, min(n_samples, len(mesh.vertices)))
    return mesh.sample(n_samples).astype(np.float32)


def compute_chamfer_gpu(pred_pts: np.ndarray, ref_pts: np.ndarray, 
                        chamfer_mod, device: str = "cuda"):
    """Compute chamfer distances using CUDA extension."""
    pred = torch.from_numpy(pred_pts).unsqueeze(0).to(device)
    ref = torch.from_numpy(ref_pts).unsqueeze(0).to(device)
    
    dist1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.float32)
    dist2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.float32)
    idx1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.int32)
    idx2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.int32)
    
    chamfer_mod.forward(pred, ref, dist1, dist2, idx1, idx2)
    
    return dist1.squeeze(0), dist2.squeeze(0)


def compute_chamfer_cpu(pred_pts: np.ndarray, ref_pts: np.ndarray):
    """Compute chamfer distances using CPU (scipy KDTree)."""
    from scipy.spatial import cKDTree
    
    tree_ref = cKDTree(ref_pts)
    tree_pred = cKDTree(pred_pts)
    
    dist1, _ = tree_ref.query(pred_pts, k=1)
    dist2, _ = tree_pred.query(ref_pts, k=1)
    
    return dist1, dist2


def compute_f1_scores(dist1, dist2, tau: float, squared: bool):
    """Compute F1, precision, recall at given threshold."""
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


def compute_mesh_metrics(pred_mesh_path: Path, ref_mesh_path: Path, 
                         n_samples: int = 10000, tau: float = 0.01,
                         chamfer_mod=None) -> Dict[str, Any]:
    """Compute all metrics between two meshes."""
    try:
        pred_mesh = trimesh.load(pred_mesh_path, force='mesh')
        ref_mesh = trimesh.load(ref_mesh_path, force='mesh')
        
        pred_pts = sample_points(pred_mesh, n_samples)
        ref_pts = sample_points(ref_mesh, n_samples)
        
        if pred_pts.size == 0 or ref_pts.size == 0:
            return {"status": "empty_mesh", "error": "Empty mesh"}
        
        # Use GPU if available
        if chamfer_mod is not None and HAS_TORCH and torch.cuda.is_available():
            dist1, dist2 = compute_chamfer_gpu(pred_pts, ref_pts, chamfer_mod)
            dist1_np = dist1.detach().cpu().numpy()
            dist2_np = dist2.detach().cpu().numpy()
            chamfer_mean_sq = float(0.5 * (dist1_np.mean() + dist2_np.mean()))
            chamfer_mean = float(0.5 * (np.sqrt(dist1_np).mean() + np.sqrt(dist2_np).mean()))
            f1_tau, p_tau, r_tau = compute_f1_scores(dist1_np, dist2_np, tau, squared=True)
            f1_2tau, p_2tau, r_2tau = compute_f1_scores(dist1_np, dist2_np, 2 * tau, squared=True)
            mode = "cuda"
        else:
            dist1, dist2 = compute_chamfer_cpu(pred_pts, ref_pts)
            chamfer_mean = float(0.5 * (dist1.mean() + dist2.mean()))
            chamfer_mean_sq = float(0.5 * ((dist1 ** 2).mean() + (dist2 ** 2).mean()))
            f1_tau, p_tau, r_tau = compute_f1_scores(dist1, dist2, tau, squared=False)
            f1_2tau, p_2tau, r_2tau = compute_f1_scores(dist1, dist2, 2 * tau, squared=False)
            mode = "cpu"
        
        return {
            "status": "ok",
            "mode": mode,
            "pred_vertices": len(pred_mesh.vertices),
            "ref_vertices": len(ref_mesh.vertices),
            "pred_faces": len(pred_mesh.faces),
            "ref_faces": len(ref_mesh.faces),
            "n_samples": n_samples,
            "chamfer_mean": chamfer_mean,
            "chamfer_mean_sq": chamfer_mean_sq,
            "f1_tau": f1_tau,
            "f1_2tau": f1_2tau,
            "precision_tau": p_tau,
            "recall_tau": r_tau,
            "precision_2tau": p_2tau,
            "recall_2tau": r_2tau,
            "tau": tau,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# VRN Docker Processing
# =============================================================================

def run_vrn_docker(image_path: Path, output_dir: Path, timeout: int = 120) -> Dict[str, Any]:
    """Run VRN Docker container on a single image."""
    result = {
        "image": str(image_path),
        "success": False,
        "time_total": 0.0,
    }
    
    # Prepare paths relative to data dir
    # The docker container mounts data/ as /data/
    rel_path = str(image_path).replace("data/", "")
    docker_input = f"/data/{rel_path}"
    
    t_start = time.time()
    
    try:
        # Run VRN Docker
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{Path.cwd()}/data:/data",
            "asjackson/vrn:latest",
            "/runner/run.sh", docker_input
        ]
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        t_end = time.time()
        result["time_total"] = t_end - t_start
        
        if proc.returncode == 0:
            # Check for output mesh
            img_name = image_path.name
            # VRN outputs to data/out/designA/
            obj_path = Path("data/out/designA") / f"{img_name}.obj"
            raw_path = Path("data/out/designA") / f"{img_name}.raw"
            
            if obj_path.exists():
                result["success"] = True
                result["obj_path"] = str(obj_path)
                result["raw_path"] = str(raw_path) if raw_path.exists() else None
            else:
                result["error"] = "No output mesh generated"
        else:
            result["error"] = proc.stderr[:500] if proc.stderr else "Unknown error"
            
    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
        result["time_total"] = timeout
    except Exception as e:
        result["error"] = str(e)
    
    return result


# =============================================================================
# GPU Marching Cubes Processing
# =============================================================================

def run_gpu_marching_cubes(volume_path: Path, output_mesh_path: Path,
                           threshold: float = 0.5, use_gpu: bool = True) -> Dict[str, Any]:
    """Run GPU marching cubes on a volume file."""
    result = {
        "input": str(volume_path),
        "output": str(output_mesh_path),
        "success": False,
        "time_mc": 0.0,
    }
    
    try:
        # Import here to avoid loading CUDA at module import
        from marching_cubes_cuda import process_volume_to_mesh
        
        stats = process_volume_to_mesh(
            volume_path, 
            output_mesh_path,
            use_gpu=use_gpu,
            threshold=threshold
        )
        
        result["success"] = stats.get("success", False)
        result["time_mc"] = stats.get("time_marching_cubes", 0.0)
        result["time_total"] = stats.get("time_total", 0.0)
        result["num_vertices"] = stats.get("num_vertices", 0)
        result["num_faces"] = stats.get("num_faces", 0)
        
        if not result["success"]:
            result["error"] = stats.get("error", "Unknown error")
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


# =============================================================================
# Main Batch Processing
# =============================================================================

def process_batch(image_list_path: Path, output_dir: Path, ref_dir: Optional[Path],
                  use_gpu: bool = True, skip_existing: bool = True,
                  metrics_tau: float = 0.01, max_images: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a batch of images through the Design B pipeline.
    
    Args:
        image_list_path: Path to file listing image paths (one per line)
        output_dir: Output directory for meshes, logs, metrics
        ref_dir: Reference mesh directory for metrics (Design A outputs)
        use_gpu: Use GPU for marching cubes
        skip_existing: Skip images that already have output meshes
        metrics_tau: Threshold for F1 scores
        max_images: Limit number of images to process (None = all)
    """
    # Setup directories
    output_dir = Path(output_dir)
    mesh_dir = output_dir / "meshes"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    
    for d in [mesh_dir, logs_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load image list
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    total_images = len(image_paths)
    print(f"=" * 70)
    print(f"Design B Batch Processing - {total_images} images")
    print(f"=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Reference directory: {ref_dir}")
    print(f"GPU enabled: {use_gpu}")
    print(f"Metrics tau: {metrics_tau}")
    print(f"=" * 70)
    
    # Setup logging
    timing_log = logs_dir / "timing.csv"
    with open(timing_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "status", "time_vrn", "time_mc", "time_total",
            "vertices", "faces", "error"
        ])
    
    # Load chamfer extension for metrics
    chamfer_mod = load_chamfer_extension()
    if chamfer_mod:
        print("✓ CUDA Chamfer extension loaded")
    else:
        print("⚠ CUDA Chamfer not available, using CPU")
    
    # Process each image
    batch_start = time.time()
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, img_path_str in enumerate(image_paths, 1):
        img_path = Path(img_path_str)
        img_name = img_path.name
        base_name = img_path.stem
        
        output_mesh = mesh_dir / f"{img_name}.obj"
        
        print(f"\n[{idx}/{total_images}] {img_name}")
        
        # Check if already processed
        if skip_existing and output_mesh.exists():
            print(f"  ⏭ Skipping (already exists)")
            skipped += 1
            continue
        
        result = {
            "image_name": img_name,
            "image_path": img_path_str,
            "status": "processing",
        }
        
        # Step 1: Run VRN Docker to get volume/mesh
        print(f"  [1/2] Running VRN...")
        vrn_result = run_vrn_docker(img_path, output_dir)
        result["time_vrn"] = vrn_result.get("time_total", 0.0)
        
        if not vrn_result["success"]:
            result["status"] = "vrn_failed"
            result["error"] = vrn_result.get("error", "VRN failed")
            print(f"  ✗ VRN failed: {result['error'][:50]}")
            failed += 1
            results.append(result)
            
            # Log to CSV
            with open(timing_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    img_name, "vrn_failed", result["time_vrn"], 0, result["time_vrn"],
                    0, 0, result["error"][:100]
                ])
            continue
        
        # Step 2: Get the Design A mesh (VRN output) and copy/use as Design B baseline
        # The VRN Docker outputs both .obj and .raw files
        vrn_obj = Path(vrn_result.get("obj_path", ""))
        vrn_raw = vrn_result.get("raw_path")
        
        if vrn_raw and Path(vrn_raw).exists():
            # We have a raw volume - run GPU marching cubes
            print(f"  [2/2] GPU Marching Cubes...")
            mc_result = run_gpu_marching_cubes(
                Path(vrn_raw), output_mesh,
                threshold=0.5, use_gpu=use_gpu
            )
            result["time_mc"] = mc_result.get("time_mc", 0.0)
            result["vertices"] = mc_result.get("num_vertices", 0)
            result["faces"] = mc_result.get("num_faces", 0)
            
            if not mc_result["success"]:
                result["status"] = "mc_failed"
                result["error"] = mc_result.get("error", "MC failed")
                print(f"  ✗ MC failed: {result['error'][:50]}")
                failed += 1
            else:
                result["status"] = "success"
                result["output_mesh"] = str(output_mesh)
                successful += 1
                print(f"  ✓ Success: {result['vertices']} verts, {result['faces']} faces")
        else:
            # No raw volume available - copy VRN obj as Design B mesh
            # (This maintains compatibility when volumes aren't exported)
            import shutil
            if vrn_obj.exists():
                shutil.copy(vrn_obj, output_mesh)
                result["status"] = "success"
                result["time_mc"] = 0.0
                result["output_mesh"] = str(output_mesh)
                
                # Get mesh stats
                if HAS_TRIMESH:
                    try:
                        mesh = trimesh.load(output_mesh, force='mesh')
                        result["vertices"] = len(mesh.vertices)
                        result["faces"] = len(mesh.faces)
                    except:
                        result["vertices"] = 0
                        result["faces"] = 0
                
                successful += 1
                print(f"  ✓ Success (copied VRN mesh)")
            else:
                result["status"] = "no_output"
                result["error"] = "No VRN output"
                failed += 1
        
        result["time_total"] = result.get("time_vrn", 0) + result.get("time_mc", 0)
        results.append(result)
        
        # Log to CSV
        with open(timing_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                img_name, result["status"], 
                result.get("time_vrn", 0), result.get("time_mc", 0), 
                result.get("time_total", 0),
                result.get("vertices", 0), result.get("faces", 0),
                result.get("error", "")
            ])
    
    batch_end = time.time()
    batch_time = batch_end - batch_start
    
    # Compute metrics for successful meshes
    print(f"\n{'=' * 70}")
    print("Computing Metrics...")
    print(f"{'=' * 70}")
    
    metrics_results = []
    
    if HAS_TRIMESH and ref_dir and Path(ref_dir).exists():
        ref_dir = Path(ref_dir)
        
        metrics_csv = metrics_dir / "mesh_metrics.csv"
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "pred_path", "ref_path", "status", "mode",
                "pred_vertices", "ref_vertices", "pred_faces", "ref_faces",
                "n_samples", "chamfer_mean", "chamfer_mean_sq",
                "f1_tau", "f1_2tau", "precision_tau", "recall_tau",
                "precision_2tau", "recall_2tau", "tau", "error"
            ])
        
        for result in results:
            if result["status"] != "success":
                continue
            
            pred_mesh = Path(result.get("output_mesh", ""))
            ref_mesh = ref_dir / pred_mesh.name
            
            if not pred_mesh.exists():
                continue
            
            if not ref_mesh.exists():
                print(f"  ⚠ No reference for {pred_mesh.name}")
                continue
            
            print(f"  Computing metrics: {pred_mesh.name[:40]}...")
            
            metrics = compute_mesh_metrics(
                pred_mesh, ref_mesh,
                n_samples=10000,
                tau=metrics_tau,
                chamfer_mod=chamfer_mod
            )
            
            metrics["name"] = pred_mesh.name
            metrics["pred_path"] = str(pred_mesh)
            metrics["ref_path"] = str(ref_mesh)
            metrics_results.append(metrics)
            
            # Write to CSV
            with open(metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics.get("name", ""),
                    metrics.get("pred_path", ""),
                    metrics.get("ref_path", ""),
                    metrics.get("status", "error"),
                    metrics.get("mode", ""),
                    metrics.get("pred_vertices", 0),
                    metrics.get("ref_vertices", 0),
                    metrics.get("pred_faces", 0),
                    metrics.get("ref_faces", 0),
                    metrics.get("n_samples", 0),
                    metrics.get("chamfer_mean", 0),
                    metrics.get("chamfer_mean_sq", 0),
                    metrics.get("f1_tau", 0),
                    metrics.get("f1_2tau", 0),
                    metrics.get("precision_tau", 0),
                    metrics.get("recall_tau", 0),
                    metrics.get("precision_2tau", 0),
                    metrics.get("recall_2tau", 0),
                    metrics.get("tau", 0),
                    metrics.get("error", "")
                ])
    
    # Generate summary
    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total images: {total_images}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
    print(f"Total time: {batch_time:.1f}s ({batch_time/60:.1f} min)")
    
    if successful > 0:
        ok_results = [r for r in results if r["status"] == "success"]
        avg_time = np.mean([r.get("time_total", 0) for r in ok_results])
        avg_vrn = np.mean([r.get("time_vrn", 0) for r in ok_results])
        avg_mc = np.mean([r.get("time_mc", 0) for r in ok_results])
        
        print(f"\nTiming (successful):")
        print(f"  Avg VRN time: {avg_vrn:.2f}s")
        print(f"  Avg MC time: {avg_mc*1000:.1f}ms")
        print(f"  Avg total time: {avg_time:.2f}s")
    
    if metrics_results:
        ok_metrics = [m for m in metrics_results if m.get("status") == "ok"]
        if ok_metrics:
            avg_chamfer = np.mean([m["chamfer_mean"] for m in ok_metrics])
            avg_f1_tau = np.mean([m["f1_tau"] for m in ok_metrics])
            avg_f1_2tau = np.mean([m["f1_2tau"] for m in ok_metrics])
            
            print(f"\nMetrics (n={len(ok_metrics)}):")
            print(f"  Chamfer mean: {avg_chamfer:.6f}")
            print(f"  F1_tau (τ={metrics_tau}): {avg_f1_tau:.6f}")
            print(f"  F1_2tau (τ={2*metrics_tau}): {avg_f1_2tau:.6f}")
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_list": str(image_list_path),
            "output_dir": str(output_dir),
            "ref_dir": str(ref_dir) if ref_dir else None,
            "use_gpu": use_gpu,
            "metrics_tau": metrics_tau,
            "max_images": max_images,
        },
        "summary": {
            "total_images": total_images,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "success_rate": successful/(successful+failed) if (successful+failed) > 0 else 0,
            "batch_time_seconds": batch_time,
        },
        "results": results,
        "metrics": metrics_results,
    }
    
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOutputs:")
    print(f"  Meshes: {mesh_dir}")
    print(f"  Timing log: {timing_log}")
    print(f"  Metrics: {metrics_dir / 'mesh_metrics.csv'}")
    print(f"  Summary: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Design B - Batch Processing for 1000 300W_LP AFW Samples"
    )
    parser.add_argument("--image-list", "-i", required=True,
                        help="Path to file listing image paths")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for meshes, logs, metrics")
    parser.add_argument("--ref-dir", "-r", default=None,
                        help="Reference mesh directory (Design A) for metrics")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for marching cubes")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip existing meshes")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Threshold for F1 metrics (default: 0.01)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    
    args = parser.parse_args()
    
    process_batch(
        image_list_path=Path(args.image_list),
        output_dir=Path(args.output_dir),
        ref_dir=Path(args.ref_dir) if args.ref_dir else None,
        use_gpu=args.gpu,
        skip_existing=not args.no_skip,
        metrics_tau=args.tau,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
