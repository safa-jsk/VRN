#!/usr/bin/env python3
"""
Design B - Demo Volume Generator
Creates synthetic volume files from Design A meshes for CUDA demonstration

Note: In a full implementation, volumes would come from VRN's internal state.
This script creates plausible volume representations for demonstrating GPU acceleration.
"""

import numpy as np
import trimesh
from pathlib import Path
import argparse
from scipy.ndimage import binary_dilation

from volume_io import load_mesh_obj, save_volume_npy


def mesh_to_volume(mesh, volume_shape=(200, 192, 192), padding=10):
    """
    Convert mesh to voxel volume for demonstration
    
    Args:
        mesh: trimesh.Trimesh object
        volume_shape: Target volume dimensions
        padding: Padding around mesh
    
    Returns:
        numpy array of shape volume_shape
    """
    # Get mesh bounds
    bounds = mesh.bounds
    mesh_size = bounds[1] - bounds[0]
    
    # Calculate voxel size
    grid_size = np.array(volume_shape) - 2 * padding
    voxel_size = np.max(mesh_size / grid_size)
    
    # Create voxel grid
    pitch = voxel_size
    
    # Voxelize the mesh
    voxels = mesh.voxelized(pitch=pitch)
    voxel_grid = voxels.matrix
    
    # Resize/pad to target shape
    current_shape = voxel_grid.shape
    volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Calculate placement (center the voxelized mesh)
    start = [(volume_shape[i] - min(current_shape[i], volume_shape[i])) // 2 
             for i in range(3)]
    end = [start[i] + min(current_shape[i], volume_shape[i]) 
           for i in range(3)]
    
    # Place voxelized mesh in volume
    src_end = [min(current_shape[i], volume_shape[i]) for i in range(3)]
    volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = \
        voxel_grid[:src_end[0], :src_end[1], :src_end[2]]
    
    # Convert binary to values similar to VRN output (0-255 range)
    volume = volume.astype(np.float32) * 255.0
    
    # Add some smoothing/dilation to make it more realistic
    if np.any(volume > 0):
        binary_vol = volume > 0
        dilated = binary_dilation(binary_vol, iterations=2)
        volume = dilated.astype(np.float32) * 200.0  # Slightly below max
    
    return volume


def create_demo_volume_from_mesh(mesh_path, output_path):
    """
    Create a demo volume file from a Design A mesh
    
    Args:
        mesh_path: Path to .obj mesh file
        output_path: Path to save .npy volume
    """
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)
    
    try:
        # Load mesh
        mesh = load_mesh_obj(mesh_path)
        
        # Convert to volume
        volume = mesh_to_volume(mesh)
        
        # Save as .npy
        save_volume_npy(volume, output_path)
        
        print(f"✓ Created demo volume: {output_path.name}")
        print(f"  Source mesh: {len(mesh.vertices)} vertices")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Value range: [{volume.min():.1f}, {volume.max():.1f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {mesh_path.name}")
        print(f"  Error: {e}")
        return False


def create_demo_volumes_batch(designA_dir, output_dir):
    """
    Create demo volumes from all Design A meshes
    
    Args:
        designA_dir: Design A mesh directory
        output_dir: Output directory for volumes
    """
    designA_dir = Path(designA_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Design B - Demo Volume Generation")
    print("=" * 60)
    print(f"Source (Design A meshes): {designA_dir}")
    print(f"Output (demo volumes):    {output_dir}")
    print("")
    print("NOTE: These are synthetic volumes created from Design A meshes")
    print("      for demonstrating GPU-accelerated marching cubes.")
    print("      In a full implementation, volumes would come from VRN.")
    print("=" * 60)
    print("")
    
    # Find all meshes
    meshes = sorted(designA_dir.glob('*.obj'))
    
    if not meshes:
        print(f"Error: No .obj files found in {designA_dir}")
        return
    
    print(f"Found {len(meshes)} Design A meshes")
    print("")
    
    # Process each mesh
    success = 0
    failed = 0
    
    for i, mesh_path in enumerate(meshes, 1):
        print(f"[{i}/{len(meshes)}] {mesh_path.stem}")
        
        output_path = output_dir / f"{mesh_path.stem}.npy"
        
        if create_demo_volume_from_mesh(mesh_path, output_path):
            success += 1
        else:
            failed += 1
        
        print("")
    
    # Summary
    print("=" * 60)
    print("Demo Volume Generation Complete")
    print("=" * 60)
    print(f"Total meshes: {len(meshes)}")
    print(f"Volumes created: {success}")
    print(f"Failed: {failed}")
    print(f"Success rate: {success/len(meshes)*100:.1f}%")
    print("")
    print(f"Output directory: {output_dir}")
    print(f"Volume count: {len(list(output_dir.glob('*.npy')))}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create demo volumes from Design A meshes for GPU demonstration'
    )
    parser.add_argument('--designA', default='data/out/designA',
                       help='Design A mesh directory')
    parser.add_argument('--output', default='data/out/designB/volumes',
                       help='Output directory for demo volumes')
    
    args = parser.parse_args()
    
    create_demo_volumes_batch(args.designA, args.output)
