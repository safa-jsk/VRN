#!/usr/bin/env python3
"""
Design B - GPU-Accelerated Marching Cubes
Implements CUDA-accelerated isosurface extraction with custom CUDA kernel

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
from pathlib import Path
from skimage.measure import marching_cubes as marching_cubes_cpu
import argparse
import sys
import os

from volume_io import (
    load_volume_npy, load_raw_volume, save_mesh_obj,
    volume_to_tensor, get_mesh_stats
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))

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
    'warmup_done': False,
}


def configure_performance_flags(cudnn_benchmark=True, tf32=True, amp=False, 
                                 compile_mode=False, warmup_iters=15):
    """
    Configure PyTorch performance flags at startup.
    
    Args:
        cudnn_benchmark: Enable cuDNN benchmark mode (default: True)
        tf32: Enable TensorFloat-32 for matmul and cuDNN (default: True)
        amp: Enable AMP autocast - NOTE: kernel executes in float32 regardless (default: False)
        compile_mode: Enable torch.compile - NOTE: limited effect on custom kernel (default: False)
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
        print("  Note: torch.compile skipped: no suitable PyTorch graph to compile (custom CUDA kernel dominates).")


def run_warmup_once(threshold=0.5, verbose=True):
    """
    Run warmup iterations once at startup to stabilize GPU performance.
    Creates a representative volume tensor and runs warmup.
    
    Args:
        threshold: Isosurface threshold
        verbose: Print warmup progress
    """
    global _PERF_CONFIG
    
    if _PERF_CONFIG['warmup_done']:
        return  # Already warmed up
    
    warmup_iters = _PERF_CONFIG['warmup_iters']
    
    if warmup_iters <= 0 or not torch.cuda.is_available():
        _PERF_CONFIG['warmup_done'] = True
        return
    
    if verbose:
        print(f"[Warmup] Running {warmup_iters} warmup iterations...")
    
    # Create representative volume (VRN shape: 200x192x192)
    warmup_volume = torch.rand(200, 192, 192, dtype=torch.float32, device='cuda') > 0.5
    warmup_volume = warmup_volume.float()
    
    t_start = time.time()
    
    for i in range(warmup_iters):
        # Run GPU marching cubes (discard results)
        _ = marching_cubes_gpu_pytorch(warmup_volume, threshold)
    
    # Synchronize after warmup
    torch.cuda.synchronize()
    
    t_end = time.time()
    
    if verbose:
        print(f"[Warmup] Complete: {t_end - t_start:.3f}s ({warmup_iters} iterations)")
    
    # Free warmup memory
    del warmup_volume
    torch.cuda.empty_cache()
    
    _PERF_CONFIG['warmup_done'] = True


def marching_cubes_gpu_pytorch(volume_tensor, threshold=0.5):
    """
    GPU-accelerated marching cubes using custom CUDA kernel.
    
    Falls back to CPU scikit-image if CUDA kernel not available.
    
    Note on AMP: The custom CUDA kernel executes in float32 regardless of AMP settings
    to preserve numerical correctness. AMP autocast is applied only to preprocessing
    tensor operations (if any), not the kernel itself.
    
    Args:
        volume_tensor: torch.Tensor on GPU (D, H, W) - boolean volume
        threshold: Isosurface threshold value (0.5 for boolean volumes)
    
    Returns:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array
    """
    if USE_CUSTOM_CUDA and torch.cuda.is_available():
        # Use custom CUDA kernel
        if not volume_tensor.is_cuda:
            volume_tensor = volume_tensor.cuda()
        
        # Ensure float32 for CUDA kernel (AMP-safe: kernel always uses float32)
        # This is critical: custom CUDA kernel expects float32, not float16
        if volume_tensor.dtype == torch.bool:
            volume_tensor = volume_tensor.float()
        elif volume_tensor.dtype != torch.float32:
            volume_tensor = volume_tensor.float()  # Convert to float32
        
        # Note: We do NOT wrap the kernel call in autocast because:
        # 1. Custom CUDA kernel expects float32 input
        # 2. AMP autocast would have no effect on custom kernel anyway
        verts, faces = marching_cubes_gpu(volume_tensor, isolevel=threshold, device='cuda')
        
        # Convert to numpy for compatibility
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
        
        return verts, faces
    else:
        # Fall back to CPU scikit-image
        volume_cpu = volume_tensor.cpu().numpy()
        
        # Convert boolean to float for scikit-image compatibility
        if volume_cpu.dtype == bool:
            volume_cpu = volume_cpu.astype(np.float32)
        
        # Use scikit-image marching cubes on CPU
        vertices, faces, normals, values = marching_cubes_cpu(
            volume_cpu, 
            level=threshold,
            spacing=(1.0, 1.0, 1.0)
        )
        
        return vertices, faces


def marching_cubes_baseline(volume, threshold=0.5):
    """
    CPU baseline marching cubes using scikit-image
    
    Args:
        volume: numpy array (D, H, W) - boolean volume
        threshold: Isosurface threshold value (0.5 for boolean volumes)
    
    Returns:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array
    """
    # Convert boolean to float for scikit-image compatibility
    if volume.dtype == bool:
        volume = volume.astype(np.float32)
    
    vertices, faces, normals, values = marching_cubes_cpu(
        volume,
        level=threshold,
        spacing=(1.0, 1.0, 1.0)
    )
    
    return vertices, faces


def process_volume_to_mesh(volume_path, output_path, 
                           use_gpu=True, threshold=0.5, 
                           timing_log=None, image_path=None):
    """
    Complete pipeline: volume -> marching cubes -> mesh export
    
    Args:
        volume_path: Path to .npy or .raw volume file
        output_path: Path to output .obj file
        use_gpu: Use GPU if available (falls back to CPU)
        threshold: Isosurface threshold (0.5 for boolean volumes)
        timing_log: Optional file to append timing data
        image_path: Optional path to input image for RGB color mapping
    
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
        'perf_flags': dict(_PERF_CONFIG),
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
            torch.cuda.synchronize()  # Ensure transfer complete before timing
            t_transfer = time.time()
            stats['time_transfer_to_gpu'] = t_transfer - t_load
            
            # GPU marching cubes with proper CUDA timing
            torch.cuda.synchronize()  # Ensure ready for timing
            t_mc_start = time.time()
            
            vertices, faces = marching_cubes_gpu_pytorch(volume_tensor, threshold)
            
            torch.cuda.synchronize()  # Ensure kernel complete before timing
            t_mc = time.time()
            stats['time_marching_cubes'] = t_mc - t_mc_start
            
        else:
            # CPU path
            if use_gpu:
                print(f"Warning: GPU requested but CUDA not available, using CPU")
            stats['device'] = 'cpu'
            
            vertices, faces = marching_cubes_baseline(volume, threshold)
            t_mc = time.time()
            stats['time_marching_cubes'] = t_mc - t_load
        
        # Save mesh with transformation and RGB colors
        # (colors are mapped AFTER transformation inside save_mesh_obj)
        mesh = save_mesh_obj(vertices, faces, output_path, 
                            apply_vrn_transform=True, 
                            image_path=image_path)
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
                          use_gpu=True, threshold=0.5,
                          pattern='*.npy', image_dir=None):
    """
    Batch process all volumes in a directory
    
    Args:
        input_dir: Directory containing .npy or .raw volume files
        output_dir: Directory for output .obj meshes
        use_gpu: Use GPU if available
        threshold: Isosurface threshold (0.5 for boolean volumes)
        pattern: Glob pattern for volume files
        image_dir: Optional directory containing input images for RGB colors
    
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
        
        # Find corresponding input image if image_dir provided
        image_path = None
        if image_dir is not None:
            image_dir_path = Path(image_dir)
            # Try common image extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                potential_img = image_dir_path / (volume_path.stem + ext)
                if potential_img.exists():
                    image_path = potential_img
                    break
        
        # Process
        stats = process_volume_to_mesh(
            volume_path, output_path,
            use_gpu=use_gpu,
            threshold=threshold,
            timing_log=timing_log,
            image_path=image_path
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
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Isosurface threshold (default: 0.5 for boolean volumes)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode (disable GPU)')
    parser.add_argument('--pattern', default='*.npy',
                       help='Glob pattern for batch mode (default: *.npy)')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory with input images for RGB color mapping')
    
    # Performance flags
    parser.add_argument('--cudnn-benchmark', type=lambda x: x.lower() == 'true',
                       default=True, metavar='BOOL',
                       help='Enable cuDNN benchmark mode (default: True)')
    parser.add_argument('--tf32', type=lambda x: x.lower() == 'true',
                       default=True, metavar='BOOL',
                       help='Enable TF32 for matmul/cuDNN (default: True)')
    parser.add_argument('--amp', type=lambda x: x.lower() == 'true',
                       default=False, metavar='BOOL',
                       help='Enable AMP autocast (default: False, kernel uses float32 regardless)')
    parser.add_argument('--compile', type=lambda x: x.lower() == 'true',
                       default=False, metavar='BOOL',
                       help='Enable torch.compile (default: False, limited effect on custom kernel)')
    parser.add_argument('--warmup-iters', type=int, default=15,
                       help='Warmup iterations before processing (default: 15)')
    parser.add_argument('--no-warmup', action='store_true',
                       help='Disable warmup iterations')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    use_gpu = not args.cpu
    
    # Configure performance flags before any CUDA work
    print("=" * 60)
    print("Design B Mesh Generation - Performance Configuration")
    print("=" * 60)
    warmup_iters = 0 if args.no_warmup else args.warmup_iters
    configure_performance_flags(
        cudnn_benchmark=args.cudnn_benchmark,
        tf32=args.tf32,
        amp=args.amp,
        compile_mode=args.compile,
        warmup_iters=warmup_iters
    )
    print("=" * 60)
    
    # Run warmup once at startup (if GPU mode and warmup enabled)
    if use_gpu and not args.no_warmup and torch.cuda.is_available():
        run_warmup_once(threshold=args.threshold, verbose=True)
    
    if input_path.is_dir():
        # Batch mode
        print("Running in BATCH mode")
        batch_process_volumes(
            input_path, output_path,
            use_gpu=use_gpu,
            threshold=args.threshold,
            pattern=args.pattern,
            image_dir=args.image_dir
        )
    else:
        # Single file mode
        print("Running in SINGLE FILE mode")
        # For single file, try to find image with same basename
        image_path = None
        if args.image_dir:
            image_dir_path = Path(args.image_dir)
            for ext in ['.jpg', '.png', '.jpeg']:
                potential_img = image_dir_path / (input_path.stem + ext)
                if potential_img.exists():
                    image_path = potential_img
                    break
        
        stats = process_volume_to_mesh(
            input_path, output_path,
            use_gpu=use_gpu,
            threshold=args.threshold,
            image_path=image_path
        )
        
        if not stats['success']:
            print(f"\nError: {stats.get('error', 'Unknown error')}")
            exit(1)
