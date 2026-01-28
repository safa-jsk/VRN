#!/usr/bin/env python3
"""
Design B - GPU-Accelerated Marching Cubes
Implements CUDA-accelerated isosurface extraction with custom CUDA kernel
"""

import numpy as np
import torch
import time
from pathlib import Path
from skimage.measure import marching_cubes as marching_cubes_cpu
import argparse
import sys
import os

from volume_io import (
    load_volume_npy, load_raw_volume, save_mesh_obj,
    volume_to_tensor, get_mesh_stats
)

# Add cuda_kernels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import custom CUDA kernel
USE_CUSTOM_CUDA = False
try:
    from cuda_kernels.cuda_marching_cubes import marching_cubes_gpu
    USE_CUSTOM_CUDA = True
    print("✓ Using custom CUDA marching cubes kernel")
except ImportError as e:
    print(f"⚠ Custom CUDA kernel not available: {e}")
    print("  Falling back to CPU scikit-image implementation")
    USE_CUSTOM_CUDA = False


def marching_cubes_gpu_pytorch(volume_tensor, threshold=10.0):
    """
    GPU-accelerated marching cubes using custom CUDA kernel.
    
    Falls back to CPU scikit-image if CUDA kernel not available.
    
    Args:
        volume_tensor: torch.Tensor on GPU (D, H, W)
        threshold: Isosurface threshold value
    
    Returns:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array
    """
    if USE_CUSTOM_CUDA and torch.cuda.is_available():
        # Use custom CUDA kernel
        if not volume_tensor.is_cuda:
            volume_tensor = volume_tensor.cuda()
        
        verts, faces = marching_cubes_gpu(volume_tensor, isolevel=threshold, device='cuda')
        
        # Convert to numpy for compatibility
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        
        return verts, faces
    else:
        # Fall back to CPU scikit-image
        volume_cpu = volume_tensor.cpu().numpy()
        
        # Use scikit-image marching cubes on CPU
    vertices, faces, normals, values = marching_cubes_cpu(
        volume_cpu, 
        level=threshold,
        spacing=(1.0, 1.0, 1.0)
    )
    
    return vertices, faces


def marching_cubes_baseline(volume, threshold=10.0):
    """
    CPU baseline marching cubes using scikit-image
    
    Args:
        volume: numpy array (D, H, W)
        threshold: Isosurface threshold value
    
    Returns:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array
    """
    vertices, faces, normals, values = marching_cubes_cpu(
        volume,
        level=threshold,
        spacing=(1.0, 1.0, 1.0)
    )
    
    return vertices, faces


def process_volume_to_mesh(volume_path, output_path, 
                           use_gpu=True, threshold=10.0, 
                           timing_log=None):
    """
    Complete pipeline: volume -> marching cubes -> mesh export
    
    Args:
        volume_path: Path to .npy or .raw volume file
        output_path: Path to output .obj file
        use_gpu: Use GPU if available (falls back to CPU)
        threshold: Isosurface threshold
        timing_log: Optional file to append timing data
    
    Returns:
        dict with processing stats
    """
    volume_path = Path(volume_path)
    output_path = Path(output_path)
    
    stats = {
        'input': str(volume_path),
        'output': str(output_path),
        'success': False,
        'device': 'cpu',
        'threshold': threshold,
    }
    
    try:
        # Load volume
        t_start = time.time()
        
        if volume_path.suffix == '.raw':
            volume = load_raw_volume(volume_path)
        elif volume_path.suffix == '.npy':
            volume = load_volume_npy(volume_path)
        else:
            raise ValueError(f"Unsupported volume format: {volume_path.suffix}")
        
        t_load = time.time()
        stats['time_load'] = t_load - t_start
        stats['volume_shape'] = volume.shape
        
        # Marching cubes
        if use_gpu and torch.cuda.is_available():
            # GPU path
            stats['device'] = torch.cuda.get_device_name(0)
            
            # Transfer to GPU
            volume_tensor = volume_to_tensor(volume, device='cuda')
            t_transfer = time.time()
            stats['time_transfer_to_gpu'] = t_transfer - t_load
            
            # GPU marching cubes
            vertices, faces = marching_cubes_gpu_pytorch(volume_tensor, threshold)
            t_mc = time.time()
            stats['time_marching_cubes'] = t_mc - t_transfer
            
        else:
            # CPU path
            if use_gpu:
                print(f"Warning: GPU requested but CUDA not available, using CPU")
            stats['device'] = 'cpu'
            
            vertices, faces = marching_cubes_baseline(volume, threshold)
            t_mc = time.time()
            stats['time_marching_cubes'] = t_mc - t_load
        
        # Save mesh
        mesh = save_mesh_obj(vertices, faces, output_path, apply_vrn_transform=True)
        t_save = time.time()
        stats['time_save'] = t_save - t_mc
        
        # Mesh statistics
        stats['num_vertices'] = len(vertices)
        stats['num_faces'] = len(faces)
        stats['time_total'] = t_save - t_start
        stats['success'] = True
        
        # Log timing if requested
        if timing_log:
            timing_log_path = Path(timing_log)
            timing_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(timing_log, 'a') as f:
                log_line = (
                    f"{volume_path.name}\t"
                    f"{stats['device']}\t"
                    f"{stats['time_marching_cubes']:.4f}\t"
                    f"{stats['time_total']:.4f}\t"
                    f"{stats['num_vertices']}\t"
                    f"{stats['num_faces']}\n"
                )
                f.write(log_line)
        
        print(f"✓ Processed: {volume_path.name}")
        print(f"  Device: {stats['device']}")
        print(f"  Marching cubes time: {stats['time_marching_cubes']:.3f}s")
        print(f"  Total time: {stats['time_total']:.3f}s")
        print(f"  Output: {output_path}")
        
    except Exception as e:
        stats['error'] = str(e)
        print(f"✗ Failed: {volume_path.name}")
        print(f"  Error: {e}")
    
    return stats


def batch_process_volumes(input_dir, output_dir, 
                          use_gpu=True, threshold=10.0,
                          pattern='*.npy'):
    """
    Batch process all volumes in a directory
    
    Args:
        input_dir: Directory containing .npy or .raw volume files
        output_dir: Directory for output .obj meshes
        use_gpu: Use GPU if available
        threshold: Isosurface threshold
        pattern: Glob pattern for volume files
    
    Returns:
        list of processing stats
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all volume files
    volume_files = sorted(input_dir.glob(pattern))
    
    if not volume_files:
        print(f"No volumes found matching {pattern} in {input_dir}")
        return []
    
    print(f"Found {len(volume_files)} volume files")
    print(f"Output directory: {output_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU: Not available (using CPU)")
    
    print("-" * 60)
    
    # Setup timing log
    timing_log = output_dir / 'marching_cubes_timing.log'
    with open(timing_log, 'w') as f:
        f.write("filename\tdevice\tmc_time\ttotal_time\tvertices\tfaces\n")
    
    # Process each volume
    all_stats = []
    t_batch_start = time.time()
    
    for i, volume_path in enumerate(volume_files, 1):
        print(f"\n[{i}/{len(volume_files)}] {volume_path.name}")
        
        # Determine output filename
        output_name = volume_path.stem + '.obj'
        output_path = output_dir / output_name
        
        # Process
        stats = process_volume_to_mesh(
            volume_path, output_path,
            use_gpu=use_gpu,
            threshold=threshold,
            timing_log=timing_log
        )
        all_stats.append(stats)
    
    t_batch_end = time.time()
    batch_time = t_batch_end - t_batch_start
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = [s for s in all_stats if s['success']]
    failed = [s for s in all_stats if not s['success']]
    
    print(f"Total processed: {len(all_stats)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(all_stats)*100:.1f}%")
    print(f"Total batch time: {batch_time:.2f}s")
    
    if successful:
        avg_mc_time = np.mean([s['time_marching_cubes'] for s in successful])
        avg_total_time = np.mean([s['time_total'] for s in successful])
        avg_vertices = np.mean([s['num_vertices'] for s in successful])
        
        print(f"\nAverage marching cubes time: {avg_mc_time:.3f}s")
        print(f"Average total time: {avg_total_time:.3f}s")
        print(f"Average vertices: {avg_vertices:.0f}")
    
    if failed:
        print(f"\nFailed files:")
        for s in failed:
            print(f"  - {Path(s['input']).name}: {s.get('error', 'Unknown error')}")
    
    print(f"\nTiming log: {timing_log}")
    
    return all_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Design B - GPU Marching Cubes for VRN volumes'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input volume file (.npy/.raw) or directory')
    parser.add_argument('--output', '-o', required=True,
                       help='Output mesh file (.obj) or directory')
    parser.add_argument('--threshold', '-t', type=float, default=10.0,
                       help='Isosurface threshold (default: 10.0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode (disable GPU)')
    parser.add_argument('--pattern', default='*.npy',
                       help='Glob pattern for batch mode (default: *.npy)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    use_gpu = not args.cpu
    
    if input_path.is_dir():
        # Batch mode
        print("Running in BATCH mode")
        batch_process_volumes(
            input_path, output_path,
            use_gpu=use_gpu,
            threshold=args.threshold,
            pattern=args.pattern
        )
    else:
        # Single file mode
        print("Running in SINGLE FILE mode")
        stats = process_volume_to_mesh(
            input_path, output_path,
            use_gpu=use_gpu,
            threshold=args.threshold
        )
        
        if not stats['success']:
            print(f"\nError: {stats.get('error', 'Unknown error')}")
            exit(1)
