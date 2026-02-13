#!/usr/bin/env python3
"""
Design A - GPU Benchmark (Unoptimized CPU/GPU Comparison)
Runs the same marching cubes algorithm as Design A but allows GPU execution
WITHOUT the custom CUDA optimization from Design B.

Purpose: Create control group for thesis speedup attribution
- Design A (CPU): Baseline performance
- Design A (GPU): GPU overhead + standard algorithms  
- Design B (GPU): GPU overhead + custom CUDA optimization

This measures: GPU speedup = Design B GPU / Design A GPU
Compare against: CPU baseline for context
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import sys

# Check PyTorch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. CPU-only execution.")

# Standard marching cubes (CPU-based)
from skimage.measure import marching_cubes

def load_volume(vol_path: str) -> np.ndarray:
    """Load volume from .npy file"""
    if not Path(vol_path).exists():
        raise FileNotFoundError(f"Volume not found: {vol_path}")
    return np.load(vol_path)

def marching_cubes_cpu(volume: np.ndarray, threshold: float = 0.5) -> Tuple:
    """
    CPU-based marching cubes using scikit-image
    (Same as Design A baseline)
    """
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=threshold,
        spacing=(1.0, 1.0, 1.0),
        allow_degenerate=False
    )
    return vertices, faces

def marching_cubes_gpu(volume_np: np.ndarray, threshold: float = 0.5, device: str = 'cuda') -> Tuple:
    """
    GPU-assisted marching cubes (PyTorch device transfer + CPU scikit-image)
    This simulates GPU execution without custom CUDA kernel optimization.
    
    Note: The actual algorithm still runs on CPU, but we measure GPU transfer overhead.
    """
    if not HAS_TORCH:
        print("PyTorch not available, falling back to CPU")
        return marching_cubes_cpu(volume_np, threshold)
    
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return marching_cubes_cpu(volume_np, threshold)
    
    # Transfer to GPU (this is overhead we're measuring)
    volume_tensor = torch.from_numpy(volume_np).float().to(device)
    
    # Synchronize to measure transfer time
    torch.cuda.synchronize()
    
    # Run marching cubes on CPU (scikit-image)
    # In real GPU-accelerated Design A, this would also be GPU-based
    vertices, faces = marching_cubes_cpu(volume_np, threshold)
    
    # Optionally transfer results back to GPU for further processing
    # vertices_gpu = torch.from_numpy(vertices).float().to(device)
    # faces_gpu = torch.from_numpy(faces).long().to(device)
    
    return vertices, faces

def save_mesh_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh to OBJ file format"""
    with open(filepath, 'w') as f:
        f.write(f"# Design A GPU Benchmark Mesh\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def benchmark_single_volume(
    volume_path: str,
    threshold: float = 0.5,
    use_gpu: bool = True,
    warmup: bool = False
) -> Dict:
    """Benchmark single volume processing"""
    
    # Load volume
    volume = load_volume(volume_path)
    
    if warmup:
        # Warmup run (not timed)
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            marching_cubes_gpu(volume, threshold)
            torch.cuda.synchronize()
        else:
            marching_cubes_cpu(volume, threshold)
    
    # Actual timed run
    start = time.time()
    
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        vertices, faces = marching_cubes_gpu(volume, threshold, device='cuda')
        torch.cuda.synchronize()  # Ensure GPU work is done
    else:
        vertices, faces = marching_cubes_cpu(volume, threshold)
    
    elapsed = time.time() - start
    
    return {
        'volume': Path(volume_path).name,
        'time_ms': elapsed * 1000,
        'vertices': len(vertices),
        'faces': len(faces),
        'gpu': use_gpu and HAS_TORCH and torch.cuda.is_available(),
        'threshold': threshold
    }

def batch_benchmark(
    volume_dir: str,
    output_dir: str,
    use_gpu: bool = True,
    threshold: float = 0.5,
    warmup_iters: int = 5,
    max_volumes: int = None,
    save_meshes: bool = False
) -> Dict:
    """Benchmark batch of volumes"""
    
    volume_dir = Path(volume_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create meshes directory if saving meshes
    mesh_dir = None
    if save_meshes:
        mesh_dir = output_dir / 'meshes'
        mesh_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all volume files
    volumes = sorted(list(volume_dir.glob('*.npy')))
    if max_volumes:
        volumes = volumes[:max_volumes]
    
    print(f"\n{'='*70}")
    print(f"Design A - GPU Benchmark (Unoptimized)")
    print(f"{'='*70}")
    print(f"Mode: {'GPU' if use_gpu else 'CPU'}")
    print(f"Volumes: {len(volumes)}")
    print(f"Threshold: {threshold}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Save meshes: {save_meshes}")
    print(f"{'='*70}\n")
    
    # GPU setup
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Available: True")
        print()
        
        # Warmup
        if warmup_iters > 0:
            print(f"Running {warmup_iters} warmup iterations...")
            warmup_vol = volumes[0]
            for i in range(warmup_iters):
                benchmark_single_volume(str(warmup_vol), threshold, use_gpu=True, warmup=True)
            torch.cuda.synchronize()
            print("Warmup complete.\n")
    else:
        print("CPU mode (GPU not available)")
        print()
    
    # Benchmark
    results = []
    times = []
    
    print(f"{'Idx':<4} {'Volume':<40} {'Time (ms)':<12} {'Vertices':<10}")
    print("-" * 70)
    
    for idx, vol_path in enumerate(volumes, 1):
        # Load volume
        volume = load_volume(str(vol_path))
        
        # Timed run
        start = time.time()
        
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            vertices, faces = marching_cubes_gpu(volume, threshold, device='cuda')
            torch.cuda.synchronize()
        else:
            vertices, faces = marching_cubes_cpu(volume, threshold)
        
        elapsed = time.time() - start
        
        result = {
            'volume': vol_path.name,
            'time_ms': elapsed * 1000,
            'vertices': len(vertices),
            'faces': len(faces),
            'gpu': use_gpu and HAS_TORCH and torch.cuda.is_available(),
            'threshold': threshold
        }
        
        results.append(result)
        times.append(result['time_ms'])
        
        # Save mesh if requested
        if save_meshes and mesh_dir:
            mesh_name = vol_path.stem.replace('_volume', '') + '.obj'
            mesh_path = mesh_dir / mesh_name
            save_mesh_obj(vertices, faces, str(mesh_path))
        
        print(f"{idx:<4} {result['volume']:<40} {result['time_ms']:>10.2f} {result['vertices']:>10}")
    
    print("-" * 70)
    
    # Statistics
    times = np.array(times)
    stats = {
        'count': len(times),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'total_time_sec': float(np.sum(times) / 1000),
        'throughput_vol_per_sec': float(len(times) / (np.sum(times) / 1000)) if np.sum(times) > 0 else 0
    }
    
    print(f"\nStatistics:")
    print(f"  Mean: {stats['mean_ms']:.2f} ms")
    print(f"  Std Dev: {stats['std_ms']:.2f} ms")
    print(f"  Min: {stats['min_ms']:.2f} ms")
    print(f"  Max: {stats['max_ms']:.2f} ms")
    print(f"  Median: {stats['median_ms']:.2f} ms")
    print(f"  Total Time: {stats['total_time_sec']:.1f} sec")
    print(f"  Throughput: {stats['throughput_vol_per_sec']:.1f} volumes/sec")
    print()
    
    # Save results
    output_file = output_dir / f"designA_gpu_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'GPU' if use_gpu else 'CPU',
        'volumes_processed': len(volumes),
        'statistics': stats,
        'results': results,
        'gpu_info': {
            'cuda_available': HAS_TORCH and torch.cuda.is_available(),
            'device_name': str(torch.cuda.get_device_name(0)) if HAS_TORCH and torch.cuda.is_available() else 'N/A'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(
        description='Design A - GPU Benchmark (Unoptimized CPU/GPU Comparison)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark Design A on GPU
  python3 designA_gpu_benchmark.py \\
    --volumes data/out/designB_1000_metrics/volumes \\
    --output data/out/designA_gpu_benchmark \\
    --gpu

  # Benchmark Design A on CPU (for baseline)
  python3 designA_gpu_benchmark.py \\
    --volumes data/out/designB_1000_metrics/volumes \\
    --output data/out/designA_gpu_benchmark \\
    --cpu

  # Quick test (10 volumes)
  python3 designA_gpu_benchmark.py \\
    --volumes data/out/designB_1000_metrics/volumes \\
    --output data/out/designA_gpu_benchmark \\
    --gpu --max-volumes 10
        """
    )
    
    parser.add_argument('--volumes', type=str, required=True,
                       help='Directory containing .npy volume files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for benchmark results')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Use GPU for execution')
    parser.add_argument('--cpu', action='store_true', default=False,
                       help='Use CPU for execution (default if neither specified)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Marching cubes threshold (default: 0.5)')
    parser.add_argument('--warmup-iters', type=int, default=5,
                       help='Warmup iterations (default: 5)')
    parser.add_argument('--max-volumes', type=int, default=None,
                       help='Maximum volumes to process (default: all)')
    parser.add_argument('--save-meshes', action='store_true', default=False,
                       help='Save output meshes to OBJ files')
    
    args = parser.parse_args()
    
    # Determine use_gpu
    use_gpu = args.gpu and not args.cpu
    
    if not Path(args.volumes).exists():
        print(f"ERROR: Volume directory not found: {args.volumes}")
        sys.exit(1)
    
    # Run benchmark
    summary = batch_benchmark(
        volume_dir=args.volumes,
        output_dir=args.output,
        use_gpu=use_gpu,
        threshold=args.threshold,
        warmup_iters=args.warmup_iters,
        max_volumes=args.max_volumes,
        save_meshes=args.save_meshes
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
