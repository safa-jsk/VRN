#!/usr/bin/env python3
"""
Design B - Performance Benchmarking
Compare GPU vs CPU marching cubes performance
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


def benchmark_single_volume(volume_path, threshold=10.0, n_runs=3):
    """
    Benchmark GPU vs CPU marching cubes on a single volume
    
    Args:
        volume_path: Path to .npy volume
        threshold: Isosurface threshold
        n_runs: Number of runs for timing (best of n)
    
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
    }
    
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
        
        gpu_times = []
        
        for i in range(n_runs):
            torch.cuda.synchronize()
            t_start = time.time()
            
            vertices_gpu, faces_gpu = marching_cubes_gpu_pytorch(volume_gpu, threshold)
            
            torch.cuda.synchronize()
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


def benchmark_batch(volume_dir, output_path, n_runs=3, max_volumes=None):
    """
    Benchmark all volumes in a directory
    
    Args:
        volume_dir: Directory containing .npy volumes
        output_path: Path to save benchmark results JSON
        n_runs: Number of runs per volume
        max_volumes: Limit number of volumes (None = all)
    
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
        result = benchmark_single_volume(vol_path, n_runs=n_runs)
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
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            max_volumes=args.max_volumes
        )
        
        # Generate plots
        plot_benchmark_results(results_path, output_dir)
