#!/usr/bin/env python3
"""
Convert all Design B volumes from float32 to boolean format
This fixes the volume type to match VRN's approach
"""

import numpy as np
from pathlib import Path
import sys

def convert_volume_to_bool(raw_path, npy_path):
    """Convert .raw volume to boolean .npy"""
    # Load as int8
    vol = np.fromfile(raw_path, dtype=np.int8)
    vol = vol.reshape((200, 192, 192))
    
    # Convert to boolean (matching VRN)
    vol_bool = vol.astype(bool)
    
    # Save as boolean npy
    np.save(npy_path, vol_bool)
    
    return np.sum(vol_bool)

def main():
    raw_dir = Path('data/out/designB/volumes_raw')
    npy_dir = Path('data/out/designB/volumes')
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} not found")
        return 1
    
    npy_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all raw volumes
    raw_files = sorted(raw_dir.glob('*.raw'))
    
    if not raw_files:
        print(f"No .raw files found in {raw_dir}")
        return 1
    
    print(f"Converting {len(raw_files)} volumes to boolean format...")
    print(f"Input:  {raw_dir}")
    print(f"Output: {npy_dir}")
    print("-" * 60)
    
    for i, raw_path in enumerate(raw_files, 1):
        npy_path = npy_dir / (raw_path.stem + '.npy')
        
        true_voxels = convert_volume_to_bool(raw_path, npy_path)
        
        print(f"[{i:2d}/{len(raw_files)}] {raw_path.name:20s} -> {npy_path.name:20s} ({true_voxels:6d} True voxels)")
    
    print("-" * 60)
    print(f"✓ Converted {len(raw_files)} volumes to boolean format")
    return 0

if __name__ == '__main__':
    sys.exit(main())
