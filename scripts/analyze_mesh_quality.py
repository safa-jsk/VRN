#!/usr/bin/env python3
"""
Simple mesh quality analysis for Design A 300W_LP output
Reports: vertex count, face count, volume, surface area
"""
import os
import trimesh
import csv
from pathlib import Path
import sys

def analyze_mesh(mesh_path):
    """Analyze a single mesh file"""
    try:
        mesh = trimesh.load(mesh_path)
        if not mesh.is_valid:
            return None
        
        return {
            'file': os.path.basename(mesh_path),
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'volume': mesh.volume if mesh.is_watertight else -1,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight
        }
    except Exception as e:
        print(f"Error analyzing {mesh_path}: {e}", file=sys.stderr)
        return None

def main():
    mesh_dir = 'data/out/designA_300w_lp'
    output_csv = 'data/out/designA_300w_lp/mesh_quality.csv'
    
    meshes = sorted(Path(mesh_dir).glob('*.obj'))
    print(f"Found {len(meshes)} meshes")
    
    results = []
    for i, mesh_path in enumerate(meshes, 1):
        print(f"[{i}/{len(meshes)}] {mesh_path.name}...", end=' ', flush=True)
        data = analyze_mesh(str(mesh_path))
        if data:
            results.append(data)
            print(f"✓ ({data['vertices']} verts, {data['faces']} faces)")
        else:
            print("✗")
    
    # Write CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nMetrics saved to {output_csv}")
        
        # Summary stats
        verts = [r['vertices'] for r in results]
        faces = [r['faces'] for r in results]
        volumes = [r['volume'] for r in results if r['volume'] > 0]
        areas = [r['surface_area'] for r in results]
        
        print(f"\nSummary Statistics ({len(results)} meshes):")
        print(f"  Vertices: min={min(verts)}, max={max(verts)}, avg={sum(verts)/len(verts):.0f}")
        print(f"  Faces: min={min(faces)}, max={max(faces)}, avg={sum(faces)/len(faces):.0f}")
        print(f"  Surface Area: min={min(areas):.0f}, max={max(areas):.0f}, avg={sum(areas)/len(areas):.0f}")
        if volumes:
            print(f"  Volume (watertight): {len(volumes)}/{len(results)}, avg={sum(volumes)/len(volumes):.0f}")

if __name__ == '__main__':
    main()
