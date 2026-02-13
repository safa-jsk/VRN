#!/usr/bin/env python3
"""
Design B - Performance Benchmarking
Compare GPU vs CPU marching cubes performance

Performance flags implemented:
- cuDNN benchmark mode
- TF32 precision (matmul + cuDNN)
- AMP autocast (optional, for completeness)
- torch.compile (optional, for completeness)
- Warmup iterations before timing
"""

import numpy as np
import torch
import time
import json
import sys
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from volume_io import load_volume_npy, volume_to_tensor
from marching_cubes_cuda import marching_cubes_baseline, marching_cubes_gpu_pytorch


# =============================================================================
# Performance Configuration
# =============================================================================

# Global config object (set by configure_performance_flags)
_PERF_CONFIG = {
    'cudnn_benchmark': True,
    'tf32': True,
    'amp': False,
    'compile': False,
    'warmup_iters': 15,
    'initialized': False,
}


def configure_performance_flags(cudnn_benchmark=True, tf32=True, amp=False, 
                                 compile_mode=False, warmup_iters=15):
    """
    Configure PyTorch performance flags at startup.
    
    Args:
        cudnn_benchmark: Enable cuDNN benchmark mode (default: True)
        tf32: Enable TensorFloat-32 for matmul and cuDNN (default: True)
        amp: Enable AMP autocast - NOTE: may have no effect on custom CUDA kernel (default: False)
        compile_mode: Enable torch.compile - NOTE: may have no effect on custom CUDA kernel (default: False)
        warmup_iters: Number of warmup iterations before timing (default: 15)
    """
    global _PERF_CONFIG
    
    _PERF_CONFIG['cudnn_benchmark'] = cudnn_benchmark
    _PERF_CONFIG['tf32'] = tf32
    _PERF_CONFIG['amp'] = amp
    _PERF_CONFIG['compile'] = compile_mode
    _PERF_CONFIG['warmup_iters'] = warmup_iters
    _PERF_CONFIG['initialized'] = True
    
    # Apply cuDNN benchmark flag
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    
    # Apply TF32 flags
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    # Print consolidated status
    print(f"[DesignB flags] cudnn_benchmark={cudnn_benchmark} tf32={tf32} "
          f"amp={amp} compile={compile_mode} warmup={warmup_iters}")
    
    # Note about effectiveness for this workload
    print("  Note: cuDNN/TF32 flags may have no effect because pipeline is custom CUDA kernel based (no convolutions).")
    if amp:
        print("  Note: AMP enabled but custom CUDA kernel executes in float32 for correctness.")
    if compile_mode:
        print("  Note: torch.compile may have limited effect on custom CUDA kernel calls.")


def run_warmup(volume_tensor, threshold=0.5, warmup_iters=None, verbose=True):
    """
    Run warmup iterations to stabilize GPU performance before timing.
    
    Args:
        volume_tensor: torch.Tensor on GPU for warmup
        threshold: Isosurface threshold
        warmup_iters: Number of warmup iterations (uses global config if None)
        verbose: Print warmup progress
    
    Returns:
        Time taken for warmup (not included in benchmarks)
    """
    if warmup_iters is None:
        warmup_iters = _PERF_CONFIG['warmup_iters']
    
    if warmup_iters <= 0:
        return 0.0
    
    if verbose:
        print(f"  Running {warmup_iters} warmup iterations...")
    
    t_warmup_start = time.time()
    
    for i in range(warmup_iters):
        # Run GPU marching cubes (discard results)
        _ = marching_cubes_gpu_pytorch(volume_tensor, threshold)
    
    # Synchronize after warmup to ensure all kernels complete
    torch.cuda.synchronize()
    
    t_warmup_end = time.time()
    warmup_time = t_warmup_end - t_warmup_start
    
    if verbose:
        print(f"  Warmup complete: {warmup_time:.3f}s ({warmup_iters} iterations)")
    
    return warmup_time


def run_cpu_warmup(volume, threshold=0.5, warmup_iters=3, verbose=True):
    """
    Run CPU warmup iterations for fair comparison (optional).
    
    Args:
        volume: numpy array volume
        threshold: Isosurface threshold
        warmup_iters: Number of CPU warmup iterations (default: 3, fewer than GPU)
        verbose: Print warmup progress
    """
    if warmup_iters <= 0:
        return 0.0
    
    if verbose:
        print(f"  Running {warmup_iters} CPU warmup iterations...")
    
    t_start = time.time()
    for _ in range(warmup_iters):
        _ = marching_cubes_baseline(volume, threshold)
    t_end = time.time()
    
    if verbose:
        print(f"  CPU warmup complete: {t_end - t_start:.3f}s")
    
    return t_end - t_start


def benchmark_single_volume(volume_path, threshold=0.5, n_runs=3, do_warmup=True):
    """
    Benchmark GPU vs CPU marching cubes on a single volume
    
    Args:
        volume_path: Path to .npy volume
        threshold: Isosurface threshold
        n_runs: Number of runs for timing (best of n)
        do_warmup: Whether to run warmup iterations (default: True)
    
    Returns:
        dict with benchmark results
    """
    print(f"\nBenchmarking: {Path(volume_path).name}")
    
    # Load volume
    volume = load_volume_npy(volume_path)
    print(f"  Volume shape: {volume.shape}")
    
    results = {
        'volume': str(volume_path),
        'shape': volume.shape,
        'threshold': threshold,
        'n_runs': n_runs,
        'warmup_iters': _PERF_CONFIG['warmup_iters'] if do_warmup else 0,
        'perf_flags': dict(_PERF_CONFIG),
    }
    
    # CPU warmup (optional, for fairness)
    if do_warmup:
        run_cpu_warmup(volume, threshold, warmup_iters=3, verbose=True)
    
    # CPU benchmark
    print(f"  Running CPU marching cubes ({n_runs} runs)...")
    cpu_times = []
    
    for i in range(n_runs):
        t_start = time.time()
        vertices_cpu, faces_cpu = marching_cubes_baseline(volume, threshold)
        t_end = time.time()
        cpu_times.append(t_end - t_start)
    
    results['cpu_time_min'] = min(cpu_times)
    results['cpu_time_avg'] = np.mean(cpu_times)
    results['cpu_time_std'] = np.std(cpu_times)
    results['cpu_vertices'] = len(vertices_cpu)
    results['cpu_faces'] = len(faces_cpu)
    
    print(f"    CPU: {results['cpu_time_min']:.4f}s (best), "
          f"{results['cpu_time_avg']:.4f}s (avg)")
    print(f"    Mesh: {results['cpu_vertices']} vertices, {results['cpu_faces']} faces")
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        print(f"  Running GPU marching cubes ({n_runs} runs)...")
        
        # Transfer to GPU
        volume_gpu = volume_to_tensor(volume, device='cuda')
        torch.cuda.synchronize()  # Ensure transfer is complete
        
        # GPU warmup (critical for accurate timing)
        if do_warmup:
            warmup_time = run_warmup(volume_gpu, threshold, verbose=True)
            results['warmup_time'] = warmup_time
        
        gpu_times = []
        
        for i in range(n_runs):
            torch.cuda.synchronize()  # Ensure previous work is done
            t_start = time.time()
            
            vertices_gpu, faces_gpu = marching_cubes_gpu_pytorch(volume_gpu, threshold)
            
            torch.cuda.synchronize()  # Ensure kernel completion before timing
            t_end = time.time()
            gpu_times.append(t_end - t_start)
        
        results['gpu_time_min'] = min(gpu_times)
        results['gpu_time_avg'] = np.mean(gpu_times)
        results['gpu_time_std'] = np.std(gpu_times)
        results['gpu_vertices'] = len(vertices_gpu)
        results['gpu_faces'] = len(faces_gpu)
        results['gpu_name'] = torch.cuda.get_device_name(0)
        results['speedup'] = results['cpu_time_min'] / results['gpu_time_min']
        
        print(f"    GPU: {results['gpu_time_min']:.4f}s (best), "
              f"{results['gpu_time_avg']:.4f}s (avg)")
        print(f"    Mesh: {results['gpu_vertices']} vertices, {results['gpu_faces']} faces")
        print(f"    Speedup: {results['speedup']:.2f}x")
        
        # GPU memory stats
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        results['gpu_memory_mb'] = mem_allocated
        
        print(f"    GPU memory: {mem_allocated:.1f} MB allocated, "
              f"{mem_reserved:.1f} MB reserved")
    else:
        print(f"  GPU not available - skipping GPU benchmark")
        results['gpu_available'] = False
    
    return results


def benchmark_batch(volume_dir, output_path, n_runs=3, max_volumes=None, do_warmup=True):
    """
    Benchmark all volumes in a directory
    
    Args:
        volume_dir: Directory containing .npy volumes
        output_path: Path to save benchmark results JSON
        n_runs: Number of runs per volume
        max_volumes: Limit number of volumes (None = all)
        do_warmup: Whether to run warmup iterations (default: True)
    
    Returns:
        list of benchmark results
    """
    volume_dir = Path(volume_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find volumes
    volumes = sorted(volume_dir.glob('*.npy'))
    
    if max_volumes:
        volumes = volumes[:max_volumes]
    
    print(f"Benchmarking {len(volumes)} volumes")
    print(f"Runs per volume: {n_runs}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available")
    
    print("=" * 60)
    
    # Benchmark each volume
    all_results = []
    
    for i, vol_path in enumerate(volumes, 1):
        print(f"\n[{i}/{len(volumes)}]")
        result = benchmark_single_volume(vol_path, n_runs=n_runs, do_warmup=do_warmup)
        all_results.append(result)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    cpu_times = [r['cpu_time_min'] for r in all_results]
    print(f"\nCPU marching cubes:")
    print(f"  Average time: {np.mean(cpu_times):.4f}s")
    print(f"  Min time: {np.min(cpu_times):.4f}s")
    print(f"  Max time: {np.max(cpu_times):.4f}s")
    print(f"  Std dev: {np.std(cpu_times):.4f}s")
    
    if torch.cuda.is_available():
        gpu_times = [r['gpu_time_min'] for r in all_results]
        speedups = [r['speedup'] for r in all_results]
        
        print(f"\nGPU marching cubes:")
        print(f"  Average time: {np.mean(gpu_times):.4f}s")
        print(f"  Min time: {np.min(gpu_times):.4f}s")
        print(f"  Max time: {np.max(gpu_times):.4f}s")
        print(f"  Std dev: {np.std(gpu_times):.4f}s")
        
        print(f"\nSpeedup (CPU/GPU):")
        print(f"  Average: {np.mean(speedups):.2f}x")
        print(f"  Min: {np.min(speedups):.2f}x")
        print(f"  Max: {np.max(speedups):.2f}x")
    
    # Save results
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'n_volumes': len(volumes),
        'n_runs': n_runs,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'perf_config': dict(_PERF_CONFIG),  # Include performance flags used
        'results': all_results,
        'summary': {
            'cpu_avg': float(np.mean(cpu_times)),
            'cpu_std': float(np.std(cpu_times)),
            'gpu_avg': float(np.mean(gpu_times)) if torch.cuda.is_available() else None,
            'gpu_std': float(np.std(gpu_times)) if torch.cuda.is_available() else None,
            'speedup_avg': float(np.mean(speedups)) if torch.cuda.is_available() else None,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def plot_benchmark_results(results_json_path, output_dir):
    """
    Generate plots from benchmark results
    
    Args:
        results_json_path: Path to benchmark results JSON
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_json_path) as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract data
    volume_names = [Path(r['volume']).stem for r in results]
    cpu_times = [r['cpu_time_min'] for r in results]
    
    # Plot 1: CPU vs GPU timing comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(volume_names))
    width = 0.35
    
    ax.bar(x - width/2, cpu_times, width, label='CPU', color='steelblue')
    
    if data['gpu_available']:
        gpu_times = [r['gpu_time_min'] for r in results]
        ax.bar(x + width/2, gpu_times, width, label='GPU', color='coral')
    
    ax.set_xlabel('Volume')
    ax.set_ylabel('Marching Cubes Time (seconds)')
    ax.set_title(f'Design B: CPU vs GPU Marching Cubes Performance\n'
                 f'GPU: {data.get("gpu_name", "N/A")}')
    ax.set_xticks(x)
    ax.set_xticklabels(volume_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'timing_comparison.png'}")
    
    # Plot 2: Speedup chart
    if data['gpu_available']:
        speedups = [r['speedup'] for r in results]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x, speedups, color='green', alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', label='No speedup')
        ax.axhline(y=np.mean(speedups), color='blue', linestyle='--', 
                   label=f'Average: {np.mean(speedups):.2f}x')
        
        ax.set_xlabel('Volume')
        ax.set_ylabel('Speedup (CPU time / GPU time)')
        ax.set_title(f'Design B: GPU Speedup over CPU\nGPU: {data["gpu_name"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(volume_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_chart.png', dpi=300)
        print(f"Saved: {output_dir / 'speedup_chart.png'}")
    
    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark GPU vs CPU marching cubes for Design B'
    )
    parser.add_argument('--volumes', required=True,
                       help='Directory containing .npy volumes')
    parser.add_argument('--output', default='data/out/designB/benchmarks',
                       help='Output directory for results')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per volume')
    parser.add_argument('--max-volumes', type=int,
                       help='Maximum number of volumes to benchmark')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots from existing results')
    parser.add_argument('--results-json',
                       help='Path to existing results JSON for plotting')
    
    # Performance flags
    parser.add_argument('--cudnn-benchmark', type=lambda x: x.lower() == 'true',
                       default=True, metavar='BOOL',
                       help='Enable cuDNN benchmark mode (default: True)')
    parser.add_argument('--tf32', type=lambda x: x.lower() == 'true',
                       default=True, metavar='BOOL',
                       help='Enable TF32 for matmul/cuDNN (default: True)')
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true',
                       default=False, metavar='BOOL',
                       help='Enable AMP autocast (default: False, limited effect on custom CUDA kernel)')
    parser.add_argument('--compile', type=lambda x: x.lower() == 'true',
                       default=False, metavar='BOOL',
                       help='Enable torch.compile (default: False, limited effect on custom CUDA kernel)')
    parser.add_argument('--warmup-iters', type=int, default=15,
                       help='Warmup iterations before timing (default: 15)')
    parser.add_argument('--no-warmup', action='store_true',
                       help='Disable warmup iterations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure performance flags before any CUDA work
    print("=" * 60)
    print("Design B Benchmark - Performance Configuration")
    print("=" * 60)
    configure_performance_flags(
        cudnn_benchmark=args.cudnn_benchmark,
        tf32=args.tf32,
        amp=args.amp,
        compile_mode=args.compile,
        warmup_iters=args.warmup_iters
    )
    print("=" * 60)
    
    if args.plot and args.results_json:
        # Plot existing results
        plot_benchmark_results(args.results_json, output_dir)
    else:
        # Run benchmark
        results_path = output_dir / 'benchmark_results.json'
        results = benchmark_batch(
            args.volumes,
            results_path,
            n_runs=args.runs,
            max_volumes=args.max_volumes,
            do_warmup=not args.no_warmup
        )
        
        # Generate plots
        plot_benchmark_results(results_path, output_dir)
