#!/usr/bin/env python3
"""
Design B - Poster Figure Generation
Generate visual comparisons between Design A and Design B for thesis poster
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import trimesh
from pathlib import Path
import json
import argparse
from PIL import Image

from volume_io import load_mesh_obj


def render_mesh_comparison(mesh_a, mesh_b, title_a='Design A', title_b='Design B'):
    """
    Render two meshes side-by-side for comparison
    
    Args:
        mesh_a: trimesh.Trimesh object
        mesh_b: trimesh.Trimesh object
        title_a: Title for mesh A
        title_b: Title for mesh B
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Create 2 subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot mesh A
    ax1.plot_trisurf(
        mesh_a.vertices[:, 0],
        mesh_a.vertices[:, 1],
        mesh_a.vertices[:, 2],
        triangles=mesh_a.faces,
        cmap='viridis',
        alpha=0.9,
        edgecolor='none'
    )
    ax1.set_title(f'{title_a}\n{len(mesh_a.vertices)} vertices')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot mesh B
    ax2.plot_trisurf(
        mesh_b.vertices[:, 0],
        mesh_b.vertices[:, 1],
        mesh_b.vertices[:, 2],
        triangles=mesh_b.faces,
        cmap='plasma',
        alpha=0.9,
        edgecolor='none'
    )
    ax2.set_title(f'{title_b}\n{len(mesh_b.vertices)} vertices')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Match viewing angles
    ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    
    plt.tight_layout()
    return fig


def generate_mesh_comparison_grid(designA_dir, designB_dir, input_dir, 
                                  output_path, n_samples=6):
    """
    Generate a grid of mesh comparisons for poster
    
    Args:
        designA_dir: Design A mesh directory
        designB_dir: Design B mesh directory
        input_dir: Input images directory
        output_path: Output image path
        n_samples: Number of samples to include
    """
    designA_dir = Path(designA_dir)
    designB_dir = Path(designB_dir)
    input_dir = Path(input_dir)
    
    # Find meshes that exist in both designs
    meshes_a = {p.name: p for p in designA_dir.glob('*.obj')}
    meshes_b = {p.name: p for p in designB_dir.glob('*.obj')}
    
    common = sorted(set(meshes_a.keys()) & set(meshes_b.keys()))
    
    if len(common) < n_samples:
        print(f"Warning: Only {len(common)} common meshes found, using all")
        n_samples = len(common)
    
    # Select samples
    samples = common[:n_samples]
    
    # Create figure grid
    fig = plt.figure(figsize=(20, n_samples * 3))
    gs = gridspec.GridSpec(n_samples, 3, width_ratios=[1, 1.5, 1.5])
    
    for i, mesh_name in enumerate(samples):
        # Load input image
        img_name = mesh_name.replace('.obj', '.jpg')
        img_path = input_dir / img_name
        
        # Input image
        ax_img = fig.add_subplot(gs[i, 0])
        if img_path.exists():
            img = Image.open(img_path)
            ax_img.imshow(img)
        ax_img.set_title(f'Input: {img_name}')
        ax_img.axis('off')
        
        # Design A mesh
        ax_a = fig.add_subplot(gs[i, 1], projection='3d')
        mesh_a = load_mesh_obj(meshes_a[mesh_name])
        ax_a.plot_trisurf(
            mesh_a.vertices[:, 0],
            mesh_a.vertices[:, 1],
            mesh_a.vertices[:, 2],
            triangles=mesh_a.faces,
            cmap='viridis',
            alpha=0.9,
            edgecolor='none'
        )
        ax_a.set_title(f'Design A (CPU)\n{len(mesh_a.vertices)} vertices')
        ax_a.axis('off')
        
        # Design B mesh
        ax_b = fig.add_subplot(gs[i, 2], projection='3d')
        mesh_b = load_mesh_obj(meshes_b[mesh_name])
        ax_b.plot_trisurf(
            mesh_b.vertices[:, 0],
            mesh_b.vertices[:, 1],
            mesh_b.vertices[:, 2],
            triangles=mesh_b.faces,
            cmap='plasma',
            alpha=0.9,
            edgecolor='none'
        )
        ax_b.set_title(f'Design B (CUDA)\n{len(mesh_b.vertices)} vertices')
        ax_b.axis('off')
        
        # Match viewing angles
        ax_b.view_init(elev=20, azim=45)
        ax_a.view_init(elev=20, azim=45)
    
    plt.suptitle('Design A (CPU) vs Design B (CUDA) - Mesh Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved mesh comparison grid: {output_path}")
    plt.close()


def generate_timing_comparison(benchmark_json, designA_metrics, output_path):
    """
    Generate timing comparison chart between Design A and Design B
    
    Args:
        benchmark_json: Path to Design B benchmark results
        designA_metrics: Path to Design A metrics
        output_path: Output image path
    """
    # Load Design B benchmarks
    with open(benchmark_json) as f:
        designB_data = json.load(f)
    
    # Load Design A metrics
    with open(designA_metrics) as f:
        lines = f.readlines()
        designA_times = []
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    designA_times.append(float(parts[1]))
                except ValueError:
                    continue
    
    # Extract Design B data
    summary = designB_data.get('summary', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Average processing time comparison
    designA_avg = np.mean(designA_times) if designA_times else 0
    designB_cpu = summary.get('cpu_avg', 0)
    designB_gpu = summary.get('gpu_avg', 0)
    
    categories = ['Design A\n(Full CPU)', 'Design B\n(MC CPU)', 'Design B\n(MC GPU)']
    times = [designA_avg, designB_cpu, designB_gpu]
    colors = ['steelblue', 'coral', 'green']
    
    bars = ax1.bar(categories, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax1.set_title('Average Processing Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Chart 2: Speedup visualization
    if designB_gpu > 0:
        speedup_vs_designA = designA_avg / designB_gpu if designB_gpu > 0 else 0
        speedup_gpu_vs_cpu = designB_cpu / designB_gpu if designB_gpu > 0 else 0
        
        speedups = [speedup_gpu_vs_cpu, speedup_vs_designA]
        labels = ['GPU vs CPU\n(MC only)', 'Design B GPU\nvs Design A']
        
        bars2 = ax2.barh(labels, speedups, color=['green', 'purple'], alpha=0.7)
        ax2.set_xlabel('Speedup Factor', fontsize=12)
        ax2.set_title('Performance Improvement', fontsize=14, fontweight='bold')
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax2.grid(axis='x', alpha=0.3)
        ax2.legend()
        
        # Add value labels
        for bar, speedup in zip(bars2, speedups):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{speedup:.2f}x',
                    ha='left', va='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add GPU info
    gpu_name = designB_data.get('gpu_name', 'N/A')
    fig.text(0.5, 0.02, f'GPU: {gpu_name}', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved timing comparison: {output_path}")
    plt.close()


def generate_pipeline_diagram(output_path):
    """
    Generate Design B pipeline architecture diagram
    
    Args:
        output_path: Output image path
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define pipeline stages
    stages = [
        {'name': 'Input\nImage', 'color': 'lightblue', 'y': 0.8},
        {'name': 'Face Detection\n& Alignment\n(CPU)', 'color': 'wheat', 'y': 0.8},
        {'name': 'VRN Volume\nRegression\n(CPU)', 'color': 'lightcoral', 'y': 0.8},
        {'name': 'Volume Export\n(.npy)', 'color': 'lightgreen', 'y': 0.8},
    ]
    
    stage2 = [
        {'name': 'Volume Load\n(GPU)', 'color': 'lightgreen', 'y': 0.4},
        {'name': 'Marching Cubes\n(CUDA)', 'color': 'gold', 'y': 0.4},
        {'name': 'Mesh Export\n(.obj)', 'color': 'plum', 'y': 0.4},
    ]
    
    # Draw Stage 1
    x_offset = 0.05
    box_width = 0.18
    box_height = 0.15
    
    ax.text(0.5, 0.95, 'Stage 1: VRN Volume Extraction (CPU)',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    for i, stage in enumerate(stages):
        x = x_offset + i * (box_width + 0.04)
        rect = Rectangle((x, stage['y'] - box_height/2), box_width, box_height,
                         facecolor=stage['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_width/2, stage['y'], stage['name'],
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(x + box_width, stage['y'], 0.03, 0,
                    head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    # Draw Stage 2
    ax.text(0.5, 0.55, 'Stage 2: GPU-Accelerated Marching Cubes',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    for i, stage in enumerate(stage2):
        x = x_offset + 0.1 + i * (box_width + 0.04)
        rect = Rectangle((x, stage['y'] - box_height/2), box_width, box_height,
                         facecolor=stage['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_width/2, stage['y'], stage['name'],
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow to next stage
        if i < len(stage2) - 1:
            ax.arrow(x + box_width, stage['y'], 0.03, 0,
                    head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    # Connection between stages
    ax.arrow(0.5, 0.72, 0, -0.15,
            head_width=0.02, head_length=0.02, fc='blue', ec='blue',
            linewidth=3, linestyle='--')
    
    # Legend
    legend_y = 0.15
    ax.text(0.1, legend_y, '● CPU Processing', fontsize=11, color='red', fontweight='bold')
    ax.text(0.4, legend_y, '● GPU Processing', fontsize=11, color='green', fontweight='bold')
    ax.text(0.7, legend_y, '● CUDA Accelerated', fontsize=11, color='blue', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Design B: Two-Stage CUDA-Accelerated Pipeline Architecture',
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved pipeline diagram: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate poster figures for Design B'
    )
    parser.add_argument('--designA', default='data/out/designA',
                       help='Design A mesh directory')
    parser.add_argument('--designB', default='data/out/designB/meshes',
                       help='Design B mesh directory')
    parser.add_argument('--input', default='data/in/aflw2000',
                       help='Input images directory')
    parser.add_argument('--benchmark', default='data/out/designB/benchmarks/benchmark_results.json',
                       help='Design B benchmark results JSON')
    parser.add_argument('--designA-metrics', default='data/out/designA/time.log',
                       help='Design A timing metrics')
    parser.add_argument('--output', default='results/poster/designB',
                       help='Output directory for poster figures')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of mesh samples for comparison grid')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Design B - Poster Figure Generation")
    print("=" * 60)
    
    # Generate mesh comparison grid
    print("\n1. Generating mesh comparison grid...")
    generate_mesh_comparison_grid(
        args.designA,
        args.designB,
        args.input,
        output_dir / 'mesh_comparisons.png',
        n_samples=args.samples
    )
    
    # Generate timing comparison
    print("\n2. Generating timing comparison...")
    if Path(args.benchmark).exists() and Path(args.designA_metrics).exists():
        generate_timing_comparison(
            args.benchmark,
            args.designA_metrics,
            output_dir / 'timing_comparison.png'
        )
    else:
        print("  Skipping: benchmark or metrics file not found")
    
    # Generate pipeline diagram
    print("\n3. Generating pipeline diagram...")
    generate_pipeline_diagram(output_dir / 'pipeline_diagram.png')
    
    print("\n" + "=" * 60)
    print("Poster figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
