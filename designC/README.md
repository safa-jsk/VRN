# Design C — PyTorch VRN with GPU Acceleration

**Objective:** Modernize VRN's neural inference to PyTorch+CUDA while preserving Design B's validated post-processing pipeline.

## Architecture

- **Input:** Face image (cropped/aligned)
- **Inference:** PyTorch volumetric CNN (GPU) → 200×192×192 occupancy volume
- **Post-processing:** Design B's CUDA marching cubes + validated transforms
- **Output:** Mesh (.obj) with RGB vertex colors

## Training Strategy

**Teacher-Student Distillation:**
- Teacher: Legacy VRN (Torch7 Docker) generates reference volumes
- Student: PyTorch model learns to predict same 200×192×192 bool volumes
- Loss: Voxel-wise BCE/focal (optionally boundary-weighted)

**Datasets:**
- Training: FaceScape multi-view (shapes 1-20) + teacher volumes from legacy VRN
- Validation: AFLW2000-3D subset (43 images) with Design B mesh quality as ground truth

## Performance Target

- **End-to-end:** <100ms (vs Design A's 2-3s)
- **Breakdown:**
  - PyTorch inference: ~50-80ms target (vs legacy VRN's 2-3s)
  - CUDA marching cubes: ~4.6ms (validated in Design B)
  - Post-processing: ~10-15ms (vertex merge + color mapping)

## Project Structure

```
designC/
├── data/
│   ├── facescape_loader.py    # PyTorch Dataset for FaceScape
│   ├── voxelizer.py            # Mesh → volume conversion
│   └── augmentation.py         # Data augmentation pipeline
├── models/
│   ├── vrn_pytorch.py          # Volumetric CNN architecture
│   └── losses.py               # BCE, focal, boundary-weighted losses
├── scripts/
│   ├── generate_teacher_volumes.sh  # Batch VRN Docker to create targets
│   ├── train.sh                     # Training launcher
│   └── benchmark.sh                 # Design A/B/C comparison
├── train.py                    # Training harness
├── pipeline.py                 # Integrated inference (PyTorch + Design B post)
├── benchmark.py                # End-to-end performance measurement
└── checkpoints/                # Saved models
```

## Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

Reuses Design B components:
- `designB/cuda_kernels/marching_cubes_cuda.py` — CUDA isosurface extraction
- `designB/python/volume_io.py` — VRN transforms (axis swap, Z×0.5, color mapping)
- `designB/python/benchmarks.py` — Verification protocol

## Quick Start

1. **Generate teacher volumes** (while images download):
   ```bash
   ./scripts/generate_teacher_volumes.sh
   ```

2. **Voxelize FaceScape shapes**:
   ```bash
   python data/voxelizer.py --input data/FaceScape/fsmview_trainset_shape_001-020/fsmview_trainset \
                             --output data/FaceScape/voxelized_shapes/
   ```

3. **Train PyTorch model**:
   ```bash
   python train.py --config configs/distillation.yaml
   ```

4. **Run benchmark**:
   ```bash
   python benchmark.py --model checkpoints/best.pth --dataset aflw2000
   ```

## Validation Protocol

Follows Design B methodology:
- 43-image AFLW2000 subset
- Best-of-3 timing with GPU synchronization
- Mesh quality checks (vertex count, bbox, visual inspection)
- Compare A/B/C outputs on identical inputs

## References

- Design A: Legacy VRN baseline (CPU)
- Design B: CUDA marching cubes (18.36x speedup on post-processing)
- Design C: Full GPU pipeline (20-30x total speedup target)
