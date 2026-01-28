#!/usr/bin/env python3
"""
Design B - Volume I/O Utilities
Handles loading VRN volume files (.raw, .npy) and mesh export
"""

import numpy as np
import torch
from pathlib import Path
import trimesh


def load_raw_volume(raw_path, shape=(200, 192, 192), dtype=np.int8):
    """
    Load VRN .raw volume file
    
    Args:
        raw_path: Path to .raw file
        shape: Volume dimensions (default: VRN's 200x192x192)
        dtype: Data type (default: int8 from VRN Torch byte)
    
    Returns:
        numpy array of shape (200, 192, 192) as boolean (matching VRN)
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Volume file not found: {raw_path}")
    
    vol = np.fromfile(raw_path, dtype=dtype)
    
    # Verify size matches expected dimensions
    expected_size = np.prod(shape)
    if vol.size != expected_size:
        raise ValueError(
            f"Volume size mismatch: expected {expected_size}, got {vol.size}"
        )
    
    vol = vol.reshape(shape)
    # Convert to boolean like VRN does (0=False, non-zero=True)
    return vol.astype(bool)


def save_volume_npy(volume, npy_path):
    """
    Save volume as .npy for faster loading
    
    Args:
        volume: numpy array
        npy_path: Output .npy path
    """
    npy_path = Path(npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, volume)
    print(f"Saved volume: {npy_path} (shape: {volume.shape})")


def load_volume_npy(npy_path):
    """
    Load volume from .npy file
    
    Args:
        npy_path: Path to .npy file
    
    Returns:
        numpy array as boolean (matching VRN)
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Volume file not found: {npy_path}")
    
    vol = np.load(npy_path)
    # Convert to boolean like VRN does
    if vol.dtype != bool:
        vol = vol.astype(np.int8).astype(bool)
    return vol


def volume_to_tensor(volume, device='cuda'):
    """
    Convert numpy volume to PyTorch tensor on specified device
    
    Args:
        volume: numpy array
        device: 'cuda' or 'cpu'
    
    Returns:
        torch.Tensor on specified device
    """
    tensor = torch.from_numpy(volume).float()
    
    if device == 'cuda' and torch.cuda.is_available():
        tensor = tensor.cuda()
    elif device == 'cuda':
        print("Warning: CUDA requested but not available, using CPU")
        tensor = tensor.cpu()
    
    return tensor


def save_mesh_obj(vertices, faces, output_path, apply_vrn_transform=True, image_path=None):
    """
    Save mesh to .obj file with VRN coordinate transform
    
    Args:
        vertices: Nx3 numpy array of vertex positions
        faces: Mx3 numpy array of triangle indices
        output_path: Output .obj path
        apply_vrn_transform: Apply VRN's coordinate system corrections
        image_path: Optional path to input image for RGB color mapping
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply VRN's coordinate transformation (from raw2obj.py)
    if apply_vrn_transform:
        vertices_transformed = vertices.copy()
        # Swap axes: (x,y,z) -> (z,y,x)
        vertices_transformed = vertices_transformed[:, [2, 1, 0]]
        # Scale Z-axis
        vertices_transformed[:, 2] *= 0.5
    else:
        vertices_transformed = vertices
    
    # Create mesh and merge vertices (matching VRN)
    mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces)
    trimesh.constants.tol.merge = 1
    mesh.merge_vertices()
    
    # Map RGB colors AFTER transformation and merging (matching VRN)
    if image_path is not None:
        try:
            from PIL import Image
            img = Image.open(image_path)
            img = img.resize((192, 192))
            img_array = np.array(img)
            
            # Map transformed vertex X,Y coordinates to image RGB (nearest neighbor)
            # After transformation, X and Y map to image coordinates
            x_img = np.clip(mesh.vertices[:, 0].astype(int), 0, 191)
            y_img = np.clip(mesh.vertices[:, 1].astype(int), 0, 191)
            
            # Extract RGB from image
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                vertex_colors = img_array[y_img, x_img, :3]
                mesh.visual.vertex_colors = vertex_colors
            elif len(img_array.shape) == 2:  # Grayscale
                gray = img_array[y_img, x_img]
                vertex_colors = np.stack([gray, gray, gray, np.ones_like(gray) * 255], axis=-1)
                mesh.visual.vertex_colors = vertex_colors
        except Exception as e:
            print(f"Warning: Could not add RGB colors: {e}")
    
    mesh.export(output_path)
    
    print(f"Saved mesh: {output_path}")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    if image_path is not None:
        print(f"  Colors: RGB")
    return mesh


def load_mesh_obj(obj_path):
    """
    Load mesh from .obj file
    
    Args:
        obj_path: Path to .obj file
    
    Returns:
        trimesh.Trimesh object
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {obj_path}")
    
    mesh = trimesh.load(obj_path)
    return mesh


def get_mesh_stats(mesh):
    """
    Compute mesh statistics for verification
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        dict with mesh statistics
    """
    stats = {
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'bbox_min': mesh.bounds[0].tolist(),
        'bbox_max': mesh.bounds[1].tolist(),
        'bbox_size': (mesh.bounds[1] - mesh.bounds[0]).tolist(),
        'volume': mesh.volume if mesh.is_volume else None,
        'is_watertight': mesh.is_watertight,
        'euler_number': mesh.euler_number,
    }
    return stats


if __name__ == '__main__':
    # Test volume I/O
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python io.py <volume.raw|volume.npy>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Load volume
    if input_path.suffix == '.raw':
        print(f"Loading .raw volume: {input_path}")
        vol = load_raw_volume(input_path)
        print(f"Volume shape: {vol.shape}")
        print(f"Value range: [{vol.min()}, {vol.max()}]")
        
        # Save as .npy
        npy_path = input_path.with_suffix('.npy')
        save_volume_npy(vol, npy_path)
        
    elif input_path.suffix == '.npy':
        print(f"Loading .npy volume: {input_path}")
        vol = load_volume_npy(input_path)
        print(f"Volume shape: {vol.shape}")
        print(f"Value range: [{vol.min()}, {vol.max()}]")
    
    # Test GPU conversion
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        tensor = volume_to_tensor(vol, device='cuda')
        print(f"Tensor on GPU: {tensor.device}, shape: {tensor.shape}")
    else:
        print("\nWarning: CUDA not available")
