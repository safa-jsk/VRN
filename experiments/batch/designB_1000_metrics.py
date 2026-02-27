#!/usr/bin/env python3
"""
Design B - Comprehensive 1000-Sample Batch Processing with Metrics
Processes 300W_LP AFW images through Design B pipeline:
1. Generate volumes from Design A meshes (or use existing)
2. Run GPU marching cubes with timing
3. Compute Chamfer, F1_tau, F1_2tau metrics
4. Generate comprehensive report

Usage:
    source vrn_env/bin/activate
    python3 scripts/designB_1000_metrics.py \
        --image-list docs/300w_lp_afw_1000.txt \
        --designA-dir data/out/designA_300w_lp \
        --output-dir data/out/designB_1000_metrics \
        --gpu
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

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
    print("Error: trimesh required for this script")
    sys.exit(1)

# Import marching cubes
try:
    from marching_cubes_cuda import (
        configure_performance_flags, 
        run_warmup_once,
        marching_cubes_gpu_pytorch,
        marching_cubes_baseline
    )
    from volume_io import (
        save_mesh_obj, save_volume_npy, load_volume_npy, 
        volume_to_tensor, get_mesh_stats
    )
    HAS_MC = True
except ImportError as e:
    print(f"Warning: Could not import marching_cubes_cuda: {e}")
    HAS_MC = False


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
    """Compute chamfer distances using CUDA extension (returns squared distances)."""
    pred = torch.from_numpy(pred_pts).unsqueeze(0).to(device)
    ref = torch.from_numpy(ref_pts).unsqueeze(0).to(device)
    
    dist1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.float32)
    dist2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.float32)
    idx1 = torch.zeros((1, pred.shape[1]), device=device, dtype=torch.int32)
    idx2 = torch.zeros((1, ref.shape[1]), device=device, dtype=torch.int32)
    
    chamfer_mod.forward(pred, ref, dist1, dist2, idx1, idx2)
    
    return dist1.squeeze(0), dist2.squeeze(0)


def compute_chamfer_pytorch_gpu(pred_pts: np.ndarray, ref_pts: np.ndarray, 
                                 device: str = "cuda", batch_size: int = 2048):
    """
    Compute chamfer distances using pure PyTorch GPU operations.
    Returns Euclidean distances (not squared) for consistency with CPU version.
    Uses batching to handle large point clouds without OOM.
    """
    pred = torch.from_numpy(pred_pts).to(device)  # (N, 3)
    ref = torch.from_numpy(ref_pts).to(device)    # (M, 3)
    
    N = pred.shape[0]
    M = ref.shape[0]
    
    # Compute dist1: for each pred point, find nearest ref point
    dist1 = torch.zeros(N, device=device)
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        pred_batch = pred[i:end_i]  # (batch, 3)
        # Compute pairwise squared distances
        diff = pred_batch.unsqueeze(1) - ref.unsqueeze(0)  # (batch, M, 3)
        sq_dists = (diff ** 2).sum(dim=2)  # (batch, M)
        dist1[i:end_i] = sq_dists.min(dim=1)[0].sqrt()  # Euclidean distance
    
    # Compute dist2: for each ref point, find nearest pred point
    dist2 = torch.zeros(M, device=device)
    for i in range(0, M, batch_size):
        end_i = min(i + batch_size, M)
        ref_batch = ref[i:end_i]  # (batch, 3)
        diff = ref_batch.unsqueeze(1) - pred.unsqueeze(0)  # (batch, N, 3)
        sq_dists = (diff ** 2).sum(dim=2)  # (batch, N)
        dist2[i:end_i] = sq_dists.min(dim=1)[0].sqrt()  # Euclidean distance
    
    return dist1.cpu().numpy(), dist2.cpu().numpy()


def compute_chamfer_cpu(pred_pts: np.ndarray, ref_pts: np.ndarray):
    """Compute chamfer distances using CPU (scipy KDTree)."""
    from scipy.spatial import cKDTree
    
    tree_ref = cKDTree(ref_pts)
    tree_pred = cKDTree(pred_pts)
    
    dist1, _ = tree_ref.query(pred_pts, k=1)  # Euclidean distance
    dist2, _ = tree_pred.query(ref_pts, k=1)
    
    return dist1, dist2


def compute_f1_scores(dist1, dist2, tau: float, squared: bool) -> Tuple[float, float, float]:
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
                         chamfer_mod=None, use_gpu: bool = True) -> Dict[str, Any]:
    """Compute all metrics between two meshes."""
    try:
        pred_mesh = trimesh.load(pred_mesh_path, force='mesh')
        ref_mesh = trimesh.load(ref_mesh_path, force='mesh')
        
        pred_pts = sample_points(pred_mesh, n_samples)
        ref_pts = sample_points(ref_mesh, n_samples)
        
        if pred_pts.size == 0 or ref_pts.size == 0:
            return {"status": "empty_mesh", "error": "Empty mesh"}
        
        # Choose computation method
        if chamfer_mod is not None and HAS_TORCH and torch.cuda.is_available():
            # Use CUDA extension (returns squared distances)
            dist1, dist2 = compute_chamfer_gpu(pred_pts, ref_pts, chamfer_mod)
            dist1_np = dist1.detach().cpu().numpy()
            dist2_np = dist2.detach().cpu().numpy()
            chamfer_mean_sq = float(0.5 * (dist1_np.mean() + dist2_np.mean()))
            chamfer_mean = float(0.5 * (np.sqrt(dist1_np).mean() + np.sqrt(dist2_np).mean()))
            f1_tau, p_tau, r_tau = compute_f1_scores(dist1_np, dist2_np, tau, squared=True)
            f1_2tau, p_2tau, r_2tau = compute_f1_scores(dist1_np, dist2_np, 2 * tau, squared=True)
            mode = "cuda_ext"
        elif use_gpu and HAS_TORCH and torch.cuda.is_available():
            # Use pure PyTorch GPU (returns Euclidean distances)
            dist1, dist2 = compute_chamfer_pytorch_gpu(pred_pts, ref_pts)
            chamfer_mean = float(0.5 * (dist1.mean() + dist2.mean()))
            chamfer_mean_sq = float(0.5 * ((dist1 ** 2).mean() + (dist2 ** 2).mean()))
            f1_tau, p_tau, r_tau = compute_f1_scores(dist1, dist2, tau, squared=False)
            f1_2tau, p_2tau, r_2tau = compute_f1_scores(dist1, dist2, 2 * tau, squared=False)
            mode = "pytorch_gpu"
        else:
            # Use CPU fallback
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
# Volume Generation from Design A Meshes
# =============================================================================

def mesh_to_volume(mesh, volume_shape=(200, 192, 192), padding=10) -> np.ndarray:
    """Convert mesh to voxel volume (simulating VRN output)."""
    bounds = mesh.bounds
    mesh_size = bounds[1] - bounds[0]
    
    grid_size = np.array(volume_shape) - 2 * padding
    voxel_size = np.max(mesh_size / grid_size)
    
    pitch = voxel_size
    voxels = mesh.voxelized(pitch=pitch)
    voxel_grid = voxels.matrix
    
    current_shape = voxel_grid.shape
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    start = [(volume_shape[i] - min(current_shape[i], volume_shape[i])) // 2 
             for i in range(3)]
    end = [start[i] + min(current_shape[i], volume_shape[i]) 
           for i in range(3)]
    
    src_end = [min(current_shape[i], volume_shape[i]) for i in range(3)]
    volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = \
        voxel_grid[:src_end[0], :src_end[1], :src_end[2]]
    
    # VRN-like output format
    volume = volume.astype(np.float32)
    
    if np.any(volume > 0):
        binary_vol = volume > 0
        dilated = binary_dilation(binary_vol, iterations=2)
        volume = dilated.astype(np.float32)
    
    return volume


def generate_volume_from_mesh(mesh_path: Path, output_path: Path) -> bool:
    """Generate volume from Design A mesh."""
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        volume = mesh_to_volume(mesh)
        np.save(output_path, volume)
        return True
    except Exception as e:
        print(f"    Error generating volume: {e}")
        return False


# =============================================================================
# GPU Marching Cubes Processing
# =============================================================================

def process_volume_gpu(volume: np.ndarray, output_path: Path, threshold: float = 0.5,
                       use_gpu: bool = True) -> Dict[str, Any]:
    """Run GPU marching cubes on volume and save mesh."""
    result = {
        "success": False,
        "time_mc": 0.0,
        "num_vertices": 0,
        "num_faces": 0,
    }
    
    try:
        t_start = time.time()
        
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            # GPU path
            volume_tensor = torch.from_numpy(volume).float().to('cuda')
            torch.cuda.synchronize()
            
            t_mc_start = time.time()
            vertices, faces = marching_cubes_gpu_pytorch(volume_tensor, threshold)
            torch.cuda.synchronize()
            t_mc_end = time.time()
            
            result["device"] = "cuda"
        else:
            # CPU path
            t_mc_start = time.time()
            vertices, faces = marching_cubes_baseline(volume, threshold)
            t_mc_end = time.time()
            result["device"] = "cpu"
        
        result["time_mc"] = t_mc_end - t_mc_start
        
        # Save mesh
        save_mesh_obj(vertices, faces, output_path, apply_vrn_transform=True)
        
        result["success"] = True
        result["num_vertices"] = len(vertices)
        result["num_faces"] = len(faces)
        result["time_total"] = time.time() - t_start
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# =============================================================================
# Main Batch Processing
# =============================================================================

def process_batch(image_list_path: Path, designA_dir: Path, output_dir: Path,
                  use_gpu: bool = True, metrics_tau: float = 0.01,
                  max_images: Optional[int] = None,
                  warmup_iters: int = 15) -> Dict[str, Any]:
    """
    Process batch through Design B pipeline with full metrics.
    
    Args:
        image_list_path: Path to file listing 1000 image paths
        designA_dir: Directory with Design A reference meshes
        output_dir: Output directory
        use_gpu: Use GPU for marching cubes
        metrics_tau: Threshold for F1 scores
        max_images: Limit number of images (None = all)
        warmup_iters: GPU warmup iterations
    """
    # Setup directories
    output_dir = Path(output_dir)
    volume_dir = output_dir / "volumes"
    mesh_dir = output_dir / "meshes"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"
    
    for d in [volume_dir, mesh_dir, logs_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load image list to get expected filenames
    with open(image_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    total_images = len(image_paths)
    
    # Find available Design A meshes
    designA_meshes = {p.name: p for p in designA_dir.glob("*.obj")}
    
    print("=" * 70)
    print("Design B - 1000 Sample Batch Processing with Metrics")
    print("=" * 70)
    print(f"Image list: {image_list_path} ({total_images} images)")
    print(f"Design A meshes: {designA_dir} ({len(designA_meshes)} available)")
    print(f"Output directory: {output_dir}")
    print(f"GPU enabled: {use_gpu}")
    print(f"Metrics tau: {metrics_tau}")
    print("=" * 70)
    
    # Configure performance flags
    if HAS_MC and use_gpu:
        configure_performance_flags(
            cudnn_benchmark=True,
            tf32=True,
            amp=False,
            compile_mode=False,
            warmup_iters=warmup_iters
        )
        
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            run_warmup_once(threshold=0.5, verbose=True)
    
    # Load chamfer extension
    chamfer_mod = load_chamfer_extension()
    if chamfer_mod:
        print("✓ CUDA Chamfer extension loaded")
    elif use_gpu and HAS_TORCH and torch.cuda.is_available():
        print("✓ Using PyTorch GPU Chamfer (pure PyTorch, no extension needed)")
    else:
        print("⚠ Using CPU Chamfer (scipy KDTree)")
    
    # Setup logging
    timing_csv = logs_dir / "timing.csv"
    with open(timing_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "status", "has_designA", "time_volume_gen", 
            "time_mc", "time_total", "vertices", "faces", "device", "error"
        ])
    
    # Process each image
    print(f"\n{'=' * 70}")
    print("Processing...")
    print("=" * 70)
    
    batch_start = time.time()
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    mc_times = []  # For timing statistics
    
    for idx, img_path_str in enumerate(image_paths, 1):
        img_path = Path(img_path_str)
        img_name = img_path.name
        mesh_name = f"{img_name}.obj"
        
        result = {
            "image_name": img_name,
            "image_path": img_path_str,
            "status": "processing",
        }
        
        # Check if Design A mesh exists
        designA_mesh = designA_meshes.get(mesh_name)
        result["has_designA"] = designA_mesh is not None
        
        if not designA_mesh:
            result["status"] = "no_designA"
            result["error"] = "No Design A mesh available"
            skipped += 1
            results.append(result)
            
            if idx % 100 == 0 or idx == total_images:
                print(f"[{idx}/{total_images}] Processed... ({successful} OK, {failed} fail, {skipped} skip)")
            continue
        
        # Step 1: Generate volume from Design A mesh
        volume_path = volume_dir / f"{img_name}.npy"
        
        t_vol_start = time.time()
        if not volume_path.exists():
            vol_success = generate_volume_from_mesh(designA_mesh, volume_path)
            if not vol_success:
                result["status"] = "volume_failed"
                result["error"] = "Volume generation failed"
                failed += 1
                results.append(result)
                continue
        
        result["time_volume_gen"] = time.time() - t_vol_start
        
        # Step 2: Run GPU marching cubes
        output_mesh = mesh_dir / mesh_name
        
        volume = np.load(volume_path)
        mc_result = process_volume_gpu(volume, output_mesh, threshold=0.5, use_gpu=use_gpu)
        
        result["time_mc"] = mc_result.get("time_mc", 0.0)
        result["time_total"] = result.get("time_volume_gen", 0) + result["time_mc"]
        result["vertices"] = mc_result.get("num_vertices", 0)
        result["faces"] = mc_result.get("num_faces", 0)
        result["device"] = mc_result.get("device", "unknown")
        
        if mc_result["success"]:
            result["status"] = "success"
            result["output_mesh"] = str(output_mesh)
            successful += 1
            mc_times.append(result["time_mc"])
        else:
            result["status"] = "mc_failed"
            result["error"] = mc_result.get("error", "MC failed")
            failed += 1
        
        results.append(result)
        
        # Log to CSV
        with open(timing_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                img_name, result["status"], result["has_designA"],
                result.get("time_volume_gen", 0), result.get("time_mc", 0),
                result.get("time_total", 0), result.get("vertices", 0),
                result.get("faces", 0), result.get("device", ""),
                result.get("error", "")
            ])
        
        # Progress update
        if idx % 50 == 0 or idx == total_images:
            print(f"[{idx}/{total_images}] {successful} OK, {failed} fail, {skipped} skip")
    
    batch_mc_time = time.time() - batch_start
    
    # Compute metrics for successful meshes
    print(f"\n{'=' * 70}")
    print("Computing Metrics...")
    print("=" * 70)
    
    metrics_csv = metrics_dir / "mesh_metrics.csv"
    metrics_results = []
    
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "pred_path", "ref_path", "status", "mode",
            "pred_vertices", "ref_vertices", "pred_faces", "ref_faces",
            "n_samples", "chamfer_mean", "chamfer_mean_sq",
            "f1_tau", "f1_2tau", "precision_tau", "recall_tau",
            "precision_2tau", "recall_2tau", "tau", "error"
        ])
    
    metrics_computed = 0
    for result in results:
        if result["status"] != "success":
            continue
        
        pred_mesh = Path(result.get("output_mesh", ""))
        mesh_name = pred_mesh.name
        ref_mesh = designA_dir / mesh_name
        
        if not pred_mesh.exists() or not ref_mesh.exists():
            continue
        
        metrics = compute_mesh_metrics(
            pred_mesh, ref_mesh,
            n_samples=10000,
            tau=metrics_tau,
            chamfer_mod=chamfer_mod,
            use_gpu=use_gpu
        )
        
        metrics["name"] = mesh_name
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
        
        metrics_computed += 1
        if metrics_computed % 50 == 0:
            print(f"  Computed {metrics_computed}/{successful} metrics...")
    
    batch_total_time = time.time() - batch_start
    
    # Generate summary
    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total images in list: {total_images}")
    print(f"Design A meshes available: {len(designA_meshes)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (no Design A): {skipped}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
    print(f"Total time: {batch_total_time:.1f}s ({batch_total_time/60:.1f} min)")
    
    if mc_times:
        print(f"\nGPU Marching Cubes Timing (n={len(mc_times)}):")
        print(f"  Mean: {np.mean(mc_times)*1000:.2f}ms")
        print(f"  Std:  {np.std(mc_times)*1000:.2f}ms")
        print(f"  Min:  {np.min(mc_times)*1000:.2f}ms")
        print(f"  Max:  {np.max(mc_times)*1000:.2f}ms")
        print(f"  Throughput: {len(mc_times)/sum(mc_times):.1f} volumes/sec")
    
    if metrics_results:
        ok_metrics = [m for m in metrics_results if m.get("status") == "ok"]
        if ok_metrics:
            avg_chamfer = np.mean([m["chamfer_mean"] for m in ok_metrics])
            std_chamfer = np.std([m["chamfer_mean"] for m in ok_metrics])
            avg_f1_tau = np.mean([m["f1_tau"] for m in ok_metrics])
            avg_f1_2tau = np.mean([m["f1_2tau"] for m in ok_metrics])
            
            print(f"\nMesh Quality Metrics (n={len(ok_metrics)}):")
            print(f"  Chamfer mean: {avg_chamfer:.6f} ± {std_chamfer:.6f}")
            print(f"  F1_tau (τ={metrics_tau}): {avg_f1_tau:.6f}")
            print(f"  F1_2tau (τ={2*metrics_tau}): {avg_f1_2tau:.6f}")
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_list": str(image_list_path),
            "designA_dir": str(designA_dir),
            "output_dir": str(output_dir),
            "use_gpu": use_gpu,
            "metrics_tau": metrics_tau,
            "warmup_iters": warmup_iters,
            "max_images": max_images,
        },
        "summary": {
            "total_images": total_images,
            "designA_available": len(designA_meshes),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "success_rate": successful/(successful+failed) if (successful+failed) > 0 else 0,
            "batch_time_seconds": batch_total_time,
        },
        "timing_stats": {
            "mc_mean_ms": float(np.mean(mc_times)*1000) if mc_times else 0,
            "mc_std_ms": float(np.std(mc_times)*1000) if mc_times else 0,
            "mc_min_ms": float(np.min(mc_times)*1000) if mc_times else 0,
            "mc_max_ms": float(np.max(mc_times)*1000) if mc_times else 0,
            "throughput_volumes_per_sec": float(len(mc_times)/sum(mc_times)) if mc_times else 0,
        },
        "metrics_stats": {},
    }
    
    if metrics_results:
        ok_metrics = [m for m in metrics_results if m.get("status") == "ok"]
        if ok_metrics:
            summary["metrics_stats"] = {
                "n_computed": len(ok_metrics),
                "chamfer_mean": float(np.mean([m["chamfer_mean"] for m in ok_metrics])),
                "chamfer_std": float(np.std([m["chamfer_mean"] for m in ok_metrics])),
                "f1_tau_mean": float(np.mean([m["f1_tau"] for m in ok_metrics])),
                "f1_2tau_mean": float(np.mean([m["f1_2tau"] for m in ok_metrics])),
                "tau": metrics_tau,
            }
    
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("Outputs:")
    print(f"  Volumes: {volume_dir}")
    print(f"  Meshes: {mesh_dir}")
    print(f"  Timing log: {timing_csv}")
    print(f"  Metrics: {metrics_csv}")
    print(f"  Summary: {summary_path}")
    print("=" * 70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Design B - 1000 Sample Batch Processing with Metrics"
    )
    parser.add_argument("--image-list", "-i", required=True,
                        help="Path to file listing image paths")
    parser.add_argument("--designA-dir", "-a", required=True,
                        help="Design A mesh directory (reference)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for marching cubes")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Threshold for F1 metrics (default: 0.01)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images")
    parser.add_argument("--warmup-iters", type=int, default=15,
                        help="GPU warmup iterations (default: 15)")
    
    args = parser.parse_args()
    
    process_batch(
        image_list_path=Path(args.image_list),
        designA_dir=Path(args.designA_dir),
        output_dir=Path(args.output_dir),
        use_gpu=args.gpu,
        metrics_tau=args.tau,
        max_images=args.max_images,
        warmup_iters=args.warmup_iters,
    )


if __name__ == "__main__":
    main()
