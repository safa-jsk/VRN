#!/usr/bin/env python3
"""
Design C – FaceScape Inference (Skeleton)

Applies the Design B GPU marching cubes pipeline to FaceScape data.
Dataset access is required; if not found, a clear message is printed.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Design C: FaceScape inference")
    parser.add_argument("--facescape_root", type=str, required=True,
                        help="Root directory of the FaceScape dataset")
    parser.add_argument("--splits_csv", type=str, required=True,
                        help="CSV file with split definitions (train/val/test)")
    parser.add_argument("--output_dir", type=str, default="artifacts/meshes_facescape",
                        help="Output directory for generated meshes")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Isosurface threshold")
    args = parser.parse_args()

    facescape_root = Path(args.facescape_root)
    if not facescape_root.exists():
        print(f"✗ FaceScape data not found at: {facescape_root}")
        print("  Please download FaceScape and provide the correct --facescape_root path.")
        print("  See: https://facescape.nju.edu.cn/")
        sys.exit(1)

    splits_csv = Path(args.splits_csv)
    if not splits_csv.exists():
        print(f"✗ Splits CSV not found at: {splits_csv}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Implement FaceScape inference pipeline
    # 1. Load split definitions from CSV
    # 2. For each subject/expression in the split:
    #    a. Load volume from facescape_root
    #    b. Run GPU marching cubes (from src.designB.pipeline)
    #    c. Save mesh to output_dir
    print(f"FaceScape inference skeleton ready.")
    print(f"  Dataset root: {facescape_root}")
    print(f"  Splits: {splits_csv}")
    print(f"  Output: {output_dir}")
    print(f"  Threshold: {args.threshold}")
    print("  → Implementation pending: add volume loading + pipeline calls.")


if __name__ == "__main__":
    main()
