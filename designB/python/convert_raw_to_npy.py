#!/usr/bin/env python3
"""
Convert VRN .raw volumes to .npy format
VRN outputs volumes as 200x192x192 uint8 .raw files
"""

import numpy as np
from pathlib import Path
import argparse


def convert_raw_to_npy(raw_path, npy_path, shape=(200, 192, 192)):
    """
    Convert VRN .raw volume to .npy
    
    Args:
        raw_path: Path to .raw file
        npy_path: Path to save .npy file
        shape: Volume dimensions (VRN default: 200x192x192)
    """
    # Read raw bytes
    with open(raw_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    
    # Reshape to 3D volume
    try:
        volume = data.reshape(shape)
    except ValueError:
        print(f"✗ Warning: {raw_path.name} has unexpected size {len(data)} bytes")
        print(f"  Expected: {np.prod(shape)} bytes for shape {shape}")
        return False
    
    # Convert to float32 for processing
    volume = volume.astype(np.float32)
    
    # Save as .npy
    np.save(npy_path, volume)
    
    return True


def batch_convert(input_dir, output_dir):
    """Convert all .raw files in directory to .npy"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted(input_dir.glob("*.raw"))
    
    if not raw_files:
        print(f"✗ No .raw files found in {input_dir}")
        return
    
    print(f"Converting {len(raw_files)} .raw files to .npy...")
    print("")
    
    success_count = 0
    for raw_file in raw_files:
        npy_file = output_dir / f"{raw_file.stem}.npy"
        
        if convert_raw_to_npy(raw_file, npy_file):
            print(f"✓ {raw_file.name} → {npy_file.name}")
            success_count += 1
        else:
            print(f"✗ Failed: {raw_file.name}")
    
    print("")
    print(f"Converted: {success_count}/{len(raw_files)} volumes")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VRN .raw volumes to .npy')
    parser.add_argument('--input', required=True, help='Input directory with .raw files')
    parser.add_argument('--output', required=True, help='Output directory for .npy files')
    parser.add_argument('--shape', nargs=3, type=int, default=[200, 192, 192],
                        help='Volume dimensions (default: 200 192 192)')
    
    args = parser.parse_args()
    
    batch_convert(args.input, args.output)
