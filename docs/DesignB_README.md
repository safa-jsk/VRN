# Design B - CUDA-Accelerated VRN Implementation

## Overview

Design B introduces **GPU acceleration** to the VRN 3D face reconstruction pipeline while preserving the baseline established in Design A. This implementation accelerates the post-processing stage (isosurface extraction) using CUDA on modern GPUs, avoiding compatibility issues with legacy Torch7 CUDA requirements.

### Key Features

- ✅ **Two-stage pipeline**: VRN volume extraction (CPU) → GPU-accelerated marching cubes
- ✅ **Modern CUDA stack**: PyTorch with CUDA 12.x, compatible with RTX 4070 SUPER
- ✅ **Reproducible baseline**: Same 50 AFLW2000 images as Design A
- ✅ **Performance metrics**: GPU vs CPU timing, speedup analysis
- ✅ **Quality verification**: Mesh comparison against Design A baseline

---

## Architecture

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: VRN Volume Extraction (CPU)                        │
├─────────────────────────────────────────────────────────────┤
│ Input Image → Face Detection → VRN Regression → Volume      │
│   (.jpg)         (dlib)          (Torch7)         (.npy)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: GPU-Accelerated Post-Processing (CUDA)             │
├─────────────────────────────────────────────────────────────┤
│ Volume Load → Marching Cubes (GPU) → Mesh Export            │
│   (.npy)         (CUDA/PyTorch)         (.obj)              │
└─────────────────────────────────────────────────────────────┘
```

### Design Rationale

**Why not full GPU VRN inference?**
- VRN was built with Torch7, requiring CUDA 7.5/8.0 + cuDNN 5.1
- These ancient versions are incompatible with modern GPUs (RTX 4070 SUPER)
- Building a CUDA-enabled Torch7 environment would be infeasible on modern hardware

**Design B approach (B1 variant):**
- Keep VRN inference on CPU (proven, reproducible)
- Accelerate **marching cubes** (the post-processing bottleneck) on GPU
- Use modern PyTorch + CUDA stack for compatibility and performance

---

## Environment Setup

### Prerequisites

- Ubuntu Linux (tested on 22.04+)
- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 4070 SUPER recommended)
- CUDA 12.x + cuDNN
- Docker (for VRN container)

### Installation

1. **Install Python dependencies**:
   ```bash
   cd cuda_post
   pip install -r requirements.txt
   ```

2. **Verify GPU availability**:
   ```bash
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```

3. **Test volume I/O**:
   ```bash
   # This will be tested after extracting volumes
   python3 cuda_post/io.py --help
   ```

---

## Usage

### Quick Start (Full Pipeline)

Process all 50 AFLW2000 images with GPU acceleration:

```bash
bash scripts/designB_run.sh data/in/aflw2000 data/out/designB
```

This runs both stages:
1. Extract volumes from VRN (CPU)
2. Generate meshes with GPU marching cubes

### Stage-by-Stage Execution

**Stage 1: Volume Extraction**
```bash
bash scripts/designB_stage1_extract_volumes.sh \
    data/in/aflw2000 \
    data/out/designB/volumes
```

**Stage 2: GPU Marching Cubes**
```bash
python3 cuda_post/marching_cubes_cuda.py \
    --input data/out/designB/volumes \
    --output data/out/designB/meshes \
    --threshold 10.0
```

**CPU-only mode** (for comparison):
```bash
python3 cuda_post/marching_cubes_cuda.py \
    --input data/out/designB/volumes \
    --output data/out/designB/meshes_cpu \
    --cpu
```

### Single Image Processing

Process a single image:

```bash
# Extract volume
docker run --rm \
    -v "$PWD/data:/data" \
    -v "/tmp/vrn_vol:/output" \
    asjackson/vrn:latest \
    /runner/run.sh /data/in/aflw2000/image00002.jpg

# Convert to .npy
python3 cuda_post/io.py /tmp/vrn_vol/image00002.raw

# GPU marching cubes
python3 cuda_post/marching_cubes_cuda.py \
    --input /tmp/vrn_vol/image00002.npy \
    --output data/out/designB/meshes/image00002.obj
```

---

## Verification & Benchmarking

### Mesh Verification

Compare Design B meshes against Design A baseline:

```bash
bash scripts/designB_verify.sh data/out/designA data/out/designB/meshes
```

Or run directly:
```bash
python3 cuda_post/verify_meshes.py \
    --designA data/out/designA \
    --designB data/out/designB/meshes \
    --output data/out/designB/verification.json
```

**Verification metrics**:
- Vertex count ratio (Design B / Design A)
- Face count ratio
- Bounding box size comparison
- Centroid distance
- Hausdorff distance (approximate)

### Performance Benchmarking

Measure GPU vs CPU marching cubes performance:

```bash
python3 cuda_post/benchmarks.py \
    --volumes data/out/designB/volumes \
    --output data/out/designB/benchmarks \
    --runs 3
```

**Outputs**:
- `benchmark_results.json`: Detailed timing data
- `timing_comparison.png`: CPU vs GPU bar chart
- `speedup_chart.png`: Speedup visualization

---

## Poster Figure Generation

Generate visual comparisons for thesis poster:

```bash
python3 cuda_post/generate_poster_figures.py \
    --designA data/out/designA \
    --designB data/out/designB/meshes \
    --input data/in/aflw2000 \
    --benchmark data/out/designB/benchmarks/benchmark_results.json \
    --output results/poster/designB \
    --samples 6
```

**Generated figures**:
- `mesh_comparisons.png`: Side-by-side mesh renderings (input image → Design A → Design B)
- `timing_comparison.png`: Processing time comparison chart
- `pipeline_diagram.png`: Two-stage architecture diagram

---

## Output Structure

```
data/out/designB/
├── volumes/              # Stage 1 output: .npy volume files
│   ├── image00002.npy
│   ├── image00008.npy
│   └── ...
├── meshes/               # Stage 2 output: .obj mesh files
│   ├── image00002.obj
│   ├── image00008.obj
│   └── ...
├── logs/
│   ├── stage1_extraction.log
│   └── stage2_marching_cubes.log
├── benchmarks/
│   ├── benchmark_results.json
│   ├── timing_comparison.png
│   └── speedup_chart.png
├── verification.json     # Mesh comparison results
└── marching_cubes_timing.log  # Per-file timing

results/poster/designB/
├── mesh_comparisons.png
├── timing_comparison.png
└── pipeline_diagram.png
```

---

## Expected Performance

### Hardware Configuration

- **GPU**: NVIDIA RTX 4070 SUPER
- **CUDA**: 12.x
- **Driver**: Latest NVIDIA drivers

### Timing Estimates (per image)

| Stage | Component | Device | Time (est.) |
|-------|-----------|--------|-------------|
| 1 | Face detection | CPU | ~1-2s |
| 1 | VRN regression | CPU | ~3-5s |
| 1 | Volume export | CPU | ~0.1s |
| 2 | Volume load | CPU/GPU | ~0.1s |
| 2 | **Marching cubes** | **GPU** | **~0.3-0.5s** |
| 2 | **Marching cubes** | **CPU** | **~2-3s** |
| 2 | Mesh export | CPU | ~0.2s |

**Expected speedup**: 
- GPU marching cubes: **4-6x faster** than CPU marching cubes
- Overall pipeline: **20-30% faster** than Design A (modest, as MC is only one stage)

### Success Rate

Design B should match Design A's success rate:
- **Expected**: 86% (43/50 successful)
- Same face detection failures as Design A

---

## Troubleshooting

### GPU Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify PyTorch CUDA installation:
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ```
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Volume Export Fails

**Symptom**: No `.npy` files in `volumes/` directory

**Solutions**:
1. Check Docker container runs successfully:
   ```bash
   docker run --rm asjackson/vrn:latest /runner/run.sh --help
   ```
2. Verify volume mount paths in [scripts/designB_stage1_extract_volumes.sh](scripts/designB_stage1_extract_volumes.sh)
3. Check disk space: `df -h`

### Mesh Quality Differs from Design A

**Symptom**: Verification shows large Hausdorff distance

**Possible causes**:
- Different marching cubes threshold (should be 10.0 for both)
- Coordinate transform mismatch (check [cuda_post/io.py](cuda_post/io.py) `apply_vrn_transform`)
- Volume loading error

**Debug**:
```bash
# Compare a single mesh
python3 cuda_post/verify_meshes.py \
    --designA data/out/designA \
    --designB data/out/designB/meshes \
    --output /tmp/debug_verify.json

# Check the output
cat /tmp/debug_verify.json | python3 -m json.tool
```

---

## Comparison: Design A vs Design B

| Aspect | Design A (Baseline) | Design B (CUDA) |
|--------|---------------------|-----------------|
| **Pipeline** | Single-stage (CPU) | Two-stage (CPU + GPU) |
| **VRN inference** | CPU (Torch7) | CPU (Torch7) |
| **Marching cubes** | CPU (mcubes) | GPU (PyTorch) |
| **Output format** | .obj | .obj |
| **Reproducibility** | 50 AFLW2000 images | Same 50 images |
| **Success rate** | 86% (43/50) | ~86% (expected) |
| **Avg. time** | 10.26s | ~8-9s (20-30% faster) |
| **CUDA story** | None | GPU acceleration demonstrated |

---

## Chapter 4 Integration

### Section 4.X: Design B (CUDA Acceleration)

**What to include**:

1. **Motivation**: Modern GPU constraints vs legacy Torch7 requirements
2. **Approach**: Two-stage pipeline with GPU post-processing
3. **Implementation**: 
   - Volume export mechanism
   - PyTorch CUDA environment setup
   - GPU marching cubes implementation
4. **Verification**:
   - Mesh quality comparison (verification.json results)
   - Performance benchmarking (speedup charts)
5. **Results**:
   - Timing comparison (Design A vs Design B)
   - GPU utilization metrics
   - Visual mesh comparisons

**Key figures for thesis**:
- [results/poster/designB/pipeline_diagram.png](results/poster/designB/pipeline_diagram.png): Two-stage architecture
- [results/poster/designB/timing_comparison.png](results/poster/designB/timing_comparison.png): Performance gains
- [results/poster/designB/mesh_comparisons.png](results/poster/designB/mesh_comparisons.png): Quality validation

---

## Implementation Status

- [x] CUDA environment setup ([cuda_post/requirements.txt](cuda_post/requirements.txt))
- [x] Volume I/O utilities ([cuda_post/io.py](cuda_post/io.py))
- [x] GPU marching cubes ([cuda_post/marching_cubes_cuda.py](cuda_post/marching_cubes_cuda.py))
- [x] Volume extraction pipeline ([scripts/designB_stage1_extract_volumes.sh](scripts/designB_stage1_extract_volumes.sh))
- [x] Two-stage runner ([scripts/designB_run.sh](scripts/designB_run.sh))
- [x] Verification tools ([cuda_post/verify_meshes.py](cuda_post/verify_meshes.py))
- [x] Benchmarking suite ([cuda_post/benchmarks.py](cuda_post/benchmarks.py))
- [x] Poster figure generation ([cuda_post/generate_poster_figures.py](cuda_post/generate_poster_figures.py))
- [x] Documentation ([docs/DesignB_README.md](docs/DesignB_README.md))

**Ready for execution**: All infrastructure in place, ready to run pipeline.

---

## Next Steps

1. **Execute pipeline**:
   ```bash
   bash scripts/designB_run.sh
   ```

2. **Run benchmarks**:
   ```bash
   python3 cuda_post/benchmarks.py --volumes data/out/designB/volumes --output data/out/designB/benchmarks
   ```

3. **Verify results**:
   ```bash
   bash scripts/designB_verify.sh
   ```

4. **Generate poster figures**:
   ```bash
   python3 cuda_post/generate_poster_figures.py
   ```

5. **Document results** in Chapter 4 of thesis

---

## References

- Design A baseline: [docs/DesignA_Implementation_Status.md](docs/DesignA_Implementation_Status.md)
- AFLW2000 subset: [docs/aflw2000_subset.txt](docs/aflw2000_subset.txt)
- VRN paper: Jackson et al., ICCV 2017
- PyTorch documentation: https://pytorch.org/docs/stable/
- RTX 4070 SUPER specs: CUDA Compute Capability 8.9

---

## Contact & Contribution

Implementation by: Ahad  
Date: January 2026  
Hardware: RTX 4070 SUPER  
CUDA Version: 12.x
