#!/usr/bin/env python3
"""
Design B - Mesh Verification
Compare Design B meshes against Design A baseline
"""

import numpy as np
import trimesh
from pathlib import Path
import argparse
import json
from datetime import datetime

from volume_io import load_mesh_obj, get_mesh_stats


def compare_meshes(mesh_a, mesh_b):
    """
    Compare two meshes and compute similarity metrics
    
    Args:
        mesh_a: trimesh.Trimesh object (Design A)
        mesh_b: trimesh.Trimesh object (Design B)
    
    Returns:
        dict with comparison metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['vertices_diff'] = abs(len(mesh_a.vertices) - len(mesh_b.vertices))
    metrics['faces_diff'] = abs(len(mesh_a.faces) - len(mesh_b.faces))
    metrics['vertices_ratio'] = len(mesh_b.vertices) / len(mesh_a.vertices) if len(mesh_a.vertices) > 0 else 0
    metrics['faces_ratio'] = len(mesh_b.faces) / len(mesh_a.faces) if len(mesh_a.faces) > 0 else 0
    
    # Bounding box comparison
    bbox_a_size = mesh_a.bounds[1] - mesh_a.bounds[0]
    bbox_b_size = mesh_b.bounds[1] - mesh_b.bounds[0]
    metrics['bbox_size_diff'] = np.linalg.norm(bbox_a_size - bbox_b_size)
    
    # Centroid comparison
    centroid_diff = np.linalg.norm(mesh_a.centroid - mesh_b.centroid)
    metrics['centroid_distance'] = float(centroid_diff)
    
    # Volume comparison (if watertight)
    if mesh_a.is_watertight and mesh_b.is_watertight:
        metrics['volume_a'] = float(mesh_a.volume)
        metrics['volume_b'] = float(mesh_b.volume)
        metrics['volume_diff'] = abs(mesh_a.volume - mesh_b.volume)
        metrics['volume_ratio'] = mesh_b.volume / mesh_a.volume if mesh_a.volume > 0 else 0
    
    # Simple Hausdorff distance approximation (sample-based)
    # Sample points from both meshes
    n_samples = min(10000, len(mesh_a.vertices), len(mesh_b.vertices))
    
    if n_samples > 100:
        samples_a = mesh_a.sample(n_samples)
        samples_b = mesh_b.sample(n_samples)
        
        # Approximate one-directional Hausdorff
        from scipy.spatial import cKDTree
        
        tree_a = cKDTree(samples_a)
        tree_b = cKDTree(samples_b)
        
        dist_b_to_a, _ = tree_a.query(samples_b)
        dist_a_to_b, _ = tree_b.query(samples_a)
        
        metrics['hausdorff_approx'] = float(max(dist_b_to_a.max(), dist_a_to_b.max()))
        metrics['mean_distance_b_to_a'] = float(dist_b_to_a.mean())
        metrics['mean_distance_a_to_b'] = float(dist_a_to_b.mean())
    
    return metrics


def verify_single_mesh(designA_path, designB_path):
    """
    Verify a single Design B mesh against Design A baseline
    
    Args:
        designA_path: Path to Design A mesh
        designB_path: Path to Design B mesh
    
    Returns:
        dict with verification results
    """
    name = Path(designB_path).stem
    print(f"\nVerifying: {name}")
    
    result = {
        'name': name,
        'designA_path': str(designA_path),
        'designB_path': str(designB_path),
    }
    
    try:
        # Load meshes
        mesh_a = load_mesh_obj(designA_path)
        mesh_b = load_mesh_obj(designB_path)
        
        # Get stats
        stats_a = get_mesh_stats(mesh_a)
        stats_b = get_mesh_stats(mesh_b)
        
        result['designA_stats'] = stats_a
        result['designB_stats'] = stats_b
        
        # Compare
        comparison = compare_meshes(mesh_a, mesh_b)
        result['comparison'] = comparison
        
        # Assessment
        result['success'] = True
        
        # Quality checks
        checks = []
        
        if 0.9 <= comparison['vertices_ratio'] <= 1.1:
            checks.append('✓ Vertex count similar')
        else:
            checks.append(f'⚠ Vertex count differs: {comparison["vertices_ratio"]:.2f}x')
        
        if 0.9 <= comparison['faces_ratio'] <= 1.1:
            checks.append('✓ Face count similar')
        else:
            checks.append(f'⚠ Face count differs: {comparison["faces_ratio"]:.2f}x')
        
        if comparison['centroid_distance'] < 10.0:
            checks.append('✓ Centroids aligned')
        else:
            checks.append(f'⚠ Centroid distance: {comparison["centroid_distance"]:.2f}')
        
        if 'hausdorff_approx' in comparison:
            if comparison['hausdorff_approx'] < 5.0:
                checks.append('✓ Geometry matches well')
            else:
                checks.append(f'⚠ Hausdorff distance: {comparison["hausdorff_approx"]:.2f}')
        
        result['checks'] = checks
        
        # Print summary
        print(f"  Design A: {stats_a['num_vertices']} vertices, {stats_a['num_faces']} faces")
        print(f"  Design B: {stats_b['num_vertices']} vertices, {stats_b['num_faces']} faces")
        print(f"  Ratio: {comparison['vertices_ratio']:.2f}x vertices, "
              f"{comparison['faces_ratio']:.2f}x faces")
        
        for check in checks:
            print(f"  {check}")
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")
    
    return result


def verify_batch(designA_dir, designB_dir, output_path):
    """
    Verify all Design B meshes against Design A baseline
    
    Args:
        designA_dir: Directory with Design A meshes
        designB_dir: Directory with Design B meshes
        output_path: Path to save verification results JSON
    """
    designA_dir = Path(designA_dir)
    designB_dir = Path(designB_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Design B Mesh Verification")
    print("=" * 60)
    print(f"Design A (baseline): {designA_dir}")
    print(f"Design B (CUDA):     {designB_dir}")
    
    # Find Design B meshes
    meshes_b = sorted(designB_dir.glob('*.obj'))
    
    print(f"\nFound {len(meshes_b)} Design B meshes")
    
    # Verify each mesh
    all_results = []
    matched = 0
    missing_baseline = 0
    
    for mesh_b_path in meshes_b:
        # Find corresponding Design A mesh
        mesh_a_path = designA_dir / mesh_b_path.name
        
        if not mesh_a_path.exists():
            print(f"\n⚠ No Design A baseline for: {mesh_b_path.name}")
            missing_baseline += 1
            continue
        
        result = verify_single_mesh(mesh_a_path, mesh_b_path)
        all_results.append(result)
        
        if result['success']:
            matched += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Design B meshes: {len(meshes_b)}")
    print(f"Matched with Design A: {matched}")
    print(f"Missing baseline: {missing_baseline}")
    
    if matched > 0:
        # Aggregate statistics
        vertex_ratios = [r['comparison']['vertices_ratio'] for r in all_results if r['success']]
        face_ratios = [r['comparison']['faces_ratio'] for r in all_results if r['success']]
        
        print(f"\nVertex count ratio (B/A):")
        print(f"  Mean: {np.mean(vertex_ratios):.3f}")
        print(f"  Std:  {np.std(vertex_ratios):.3f}")
        print(f"  Range: [{np.min(vertex_ratios):.3f}, {np.max(vertex_ratios):.3f}]")
        
        print(f"\nFace count ratio (B/A):")
        print(f"  Mean: {np.mean(face_ratios):.3f}")
        print(f"  Std:  {np.std(face_ratios):.3f}")
        print(f"  Range: [{np.min(face_ratios):.3f}, {np.max(face_ratios):.3f}]")
        
        if any('hausdorff_approx' in r['comparison'] for r in all_results if r['success']):
            hausdorff_dists = [r['comparison']['hausdorff_approx'] 
                              for r in all_results 
                              if r['success'] and 'hausdorff_approx' in r['comparison']]
            print(f"\nHausdorff distance (approx):")
            print(f"  Mean: {np.mean(hausdorff_dists):.3f}")
            print(f"  Max:  {np.max(hausdorff_dists):.3f}")
    
    # Save results
    verification_data = {
        'timestamp': datetime.now().isoformat(),
        'designA_dir': str(designA_dir),
        'designB_dir': str(designB_dir),
        'total_meshes': len(meshes_b),
        'matched': matched,
        'missing_baseline': missing_baseline,
        'results': all_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(verification_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify Design B meshes against Design A baseline'
    )
    parser.add_argument('--designA', default='data/out/designA',
                       help='Design A mesh directory')
    parser.add_argument('--designB', default='data/out/designB/meshes',
                       help='Design B mesh directory')
    parser.add_argument('--output', default='data/out/designB/verification.json',
                       help='Output JSON path for verification results')
    
    args = parser.parse_args()
    
    verify_batch(args.designA, args.designB, args.output)
