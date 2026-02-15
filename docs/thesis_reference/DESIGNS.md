# VRN Design Variants - Thesis Documentation

**Purpose:** Document all four design variants (A, A_GPU, B, C) with exact entrypoints, configuration flags, expected outputs, and timing measurement locations.

---

## Design A: Legacy CPU Baseline

### Overview

- **Type:** Baseline reference implementation
- **Framework:** Torch7 (legacy, CPU-only)
- **Environment:** Docker container (`asjackson/vrn:latest`)
- **Primary Goal:** Establish reproducible baseline for comparison
- **CAMFM Stages:** None (reference baseline)

### Entrypoint Script

**Primary:** `run.sh`

```bash
#!/usr/bin/env bash
# Line 1-93 in run.sh

# Main execution
docker run --rm \
  -v "$PWD/data:/data" \
  asjackson/vrn:latest \
  /runner/run.sh /data/in/<image>.jpg
```

**Alternative (Batch):** `scripts/batch_process_aflw2000.sh`

```bash
#!/usr/bin/env bash
# Lines 1-150

# Batch processing on AFLW2000 dataset
for img in "${image_list[@]}"; do
  docker run --rm \
    -v "$PWD/data:/data" \
    asjackson/vrn:latest \
    /runner/run.sh "$img"
done
```

### Configuration Flags

| Flag                   | Default           | Purpose                         | Location    |
| ---------------------- | ----------------- | ------------------------------- | ----------- |
| `INPUT`                | `examples/`       | Input image directory           | `run.sh` L3 |
| `OUTPUT`               | `output/`         | Raw volume output directory     | `run.sh` L4 |
| `VRN_MODEL`            | `vrn-unguided.t7` | VRN model weights               | `run.sh` L5 |
| `CUDA_VISIBLE_DEVICES` | `0`               | GPU number (unused in CPU mode) | `run.sh` L6 |

**Face Alignment Config (inside container):**

- Model: `2D-FAN-300W.t7` (2D face alignment network)
- Mode: `generate` (generate landmarks)
- Device: `gpu` (but runs on CPU in container)
- Output format: `txt` (landmark coordinates)

**VRN Inference Config (inside container):**

- Model: `vrn-unguided.t7`
- Input size: 192×192 RGB
- Output size: 200×192×192 voxels
- Device: CPU (Torch7 float tensors)

### Expected Outputs

**Per-Image Outputs:**

```
data/in/image00002.jpg              # Original input
data/in/image00002.jpg.crop.jpg     # Cropped/aligned face
data/in/image00002.jpg.txt          # Facial landmarks
data/in/image00002.jpg.obj          # 3D mesh output
```

**Batch Processing Outputs:**

```
data/out/designA/
├── image00002.obj                  # Mesh files (43 successful)
├── image00004.obj
├── ...
├── RESULTS_SUMMARY.txt             # Batch statistics
└── time.log                        # Per-image timing

data/out/designA_300w_lp/
├── AFW_134212_1_0.obj              # 300W_LP meshes (468 successful)
├── ...
├── DesignA_Evaluation_Summary.md   # Comprehensive evaluation
├── DesignA_Metrics_Summary.md      # Detailed metrics
└── DesignA_Metrics.csv             # Sample data
```

### Timing Measurement Locations

**High-Level Timing (Batch Script):**

- **Location:** `scripts/batch_process_aflw2000.sh` lines 87-99
- **Measurement:** Wall-clock time per image (entire pipeline)
- **Logging:** Written to `data/out/designA/time.log`
- **Format:** `image00002.jpg: 8.234 seconds`

**Low-Level Timing (inside Docker container):**

- Face detection: Not logged separately
- VRN inference: Not logged separately
- Marching cubes: Not logged separately
- Total time: Printed to stdout by container

**Metrics Computation Timing:**

- **Location:** `scripts/designA_mesh_metrics.py` lines 182-192
- **Measurement:** Chamfer distance computation time
- **Output:** Included in metrics CSV

### Performance Characteristics

**AFLW2000-3D Dataset (1000 images tested):**

- Success rate: 4.3% (43/1000)
- Average time per successful mesh: ~6-8 seconds
- Average time per failed detection: ~2 seconds
- Total batch time: ~400 minutes (6.7 hours)

**300W_LP Dataset (1000 images tested):**

- Success rate: 46.8% (468/1000)
- Average time per successful mesh: ~15 seconds
- Average time per failed detection: ~2 seconds
- Total batch time: 116 minutes 42 seconds

### Limitations

- **CPU-only:** No GPU acceleration
- **Legacy framework:** Torch7 (difficult to extend)
- **Detector limitation:** dlib HOG frontal face only (±45° max)
- **No metrics logging:** Requires post-processing script

---

## Design A_GPU: Simple GPU Enablement

### Overview

- **Type:** Conceptual GPU port (not fully implemented)
- **Framework:** PyTorch 2.x with GPU tensor ops
- **Primary Goal:** Demonstrate basic GPU residency without custom kernels
- **CAMFM Stages:** CAMFM.A2a_GPU_RESIDENCY (partial)

### Implementation Status

- **Current:** Conceptual only (see `docs/DesignA_GPU_Evaluation_Summary.md`)
- **Proposed Entrypoint:** `scripts/designA_gpu_benchmark.py`
- **Key Change:** Use `torch.tensor(..., device='cuda')` for VRN inference
- **Expected Speedup:** 2-3× (GPU tensor ops only)

### Why Not Fully Implemented

- Limited benefit without custom kernels
- VRN model architecture not optimized for GPU
- Design B provides superior acceleration (18.36× with custom CUDA)
- Effort better spent on Design B and C

### Hypothetical Configuration

```python
# Proposed config (not implemented)
config = {
    'device': 'cuda',
    'model_path': 'vrn-unguided.pth',  # Converted PyTorch weights
    'batch_size': 1,
    'use_cudnn': True,
    'amp': False,  # Not stable for VRN architecture
}
```

---

## Design B: CUDA-Optimized Pipeline

### Overview

- **Type:** GPU-accelerated with custom CUDA marching cubes
- **Framework:** PyTorch 2.1.0 + CUDA 11.8
- **Environment:** Host system (RTX 2050, RTX 4070 SUPER tested)
- **Primary Goal:** Demonstrate measurable CUDA acceleration (18.36× speedup)
- **CAMFM Stages:** A2a, A2b, A2c, A3, A5 (full implementation)

### Entrypoint Scripts

**1. Single Volume Processing:**

```bash
# File: designB/python/marching_cubes_cuda.py
# Lines: 335-385

python3 designB/python/marching_cubes_cuda.py \
  --input data/out/designB/volumes/image00002.npy \
  --output data/out/designB/meshes/image00002.obj \
  --threshold 0.5
```

**2. Batch Processing:**

```bash
# File: designB/python/marching_cubes_cuda.py
# Lines: 223-320

python3 designB/python/marching_cubes_cuda.py \
  --input data/out/designB/volumes \
  --output data/out/designB/meshes \
  --threshold 0.5 \
  --pattern '*.npy'
```

**3. Benchmark/Evaluation:**

```bash
# File: designB/python/benchmarks.py
# Lines: 287-318

python3 designB/python/benchmarks.py \
  --volumes data/out/designB/volumes \
  --output data/out/designB/benchmarks_cuda \
  --runs 3 \
  --plot

# Or use wrapper script
./designB/scripts/run_benchmarks.sh
```

### Configuration Flags

**Performance Configuration:**

- **Location:** `designB/python/benchmarks.py` lines 38-86
- **Function:** `configure_performance_flags()`

| Flag              | Default | Purpose                               | CAMFM Stage        |
| ----------------- | ------- | ------------------------------------- | ------------------ |
| `cudnn_benchmark` | `True`  | Enable cuDNN autotuner                | A2b_STEADY_STATE   |
| `tf32`            | `True`  | Enable TensorFloat-32                 | A2b_STEADY_STATE   |
| `amp`             | `False` | AMP autocast (N/A for custom kernel)  | A2d_OPTIONAL_ACCEL |
| `compile_mode`    | `False` | torch.compile (N/A for custom kernel) | A2d_OPTIONAL_ACCEL |
| `warmup_iters`    | `15`    | Warmup iterations before timing       | A2b_STEADY_STATE   |

**Applied Settings:**

```python
# designB/python/benchmarks.py lines 73-79
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Marching Cubes Parameters:**

- **Location:** `designB/cuda_kernels/cuda_marching_cubes.py` lines 22-79

| Parameter       | Default             | Purpose                   | CAMFM Stage       |
| --------------- | ------------------- | ------------------------- | ----------------- |
| `threshold`     | `0.5`               | Isosurface threshold      | N/A               |
| `device`        | `'cuda'`            | Target device             | A2a_GPU_RESIDENCY |
| `max_vertices`  | `dimX*dimY*dimZ*15` | Pre-allocated buffer size | A2c_MEM_LAYOUT    |
| `max_triangles` | `dimX*dimY*dimZ*5`  | Pre-allocated buffer size | A2c_MEM_LAYOUT    |

### Expected Outputs

**Mesh Files:**

```
data/out/designB/meshes/
├── image00002.obj                  # 3D mesh (Wavefront OBJ)
├── image00004.obj
├── ...
└── (43 total for AFLW2000)
```

**Benchmark Results:**

```
data/out/designB/benchmarks_cuda/
├── benchmark_results.json          # Detailed timing data
├── timing_comparison.png           # CPU vs GPU plot
└── speedup_chart.png               # Speedup visualization
```

**Benchmark JSON Structure:**

```json
{
  "image00002.npy": {
    "cpu_times": [0.1234, 0.1245, 0.1238],
    "gpu_times": [0.0067, 0.0065, 0.0066],
    "cpu_avg": 0.1239,
    "gpu_avg": 0.0066,
    "speedup": 18.77,
    "std_dev_cpu": 0.0005,
    "std_dev_gpu": 0.0001
  },
  // ... 42 more entries
  "overall": {
    "avg_speedup": 18.36,
    "median_speedup": 18.24,
    "min_speedup": 16.89,
    "max_speedup": 19.78
  }
}
```

### Timing Measurement Locations

**Warmup Phase:**

- **Location:** `designB/python/benchmarks.py` lines 115-138
- **Function:** `run_warmup()`
- **Purpose:** CAMFM.A2b_STEADY_STATE compliance
- **Implementation:**
  ```python
  for i in range(warmup_iters):
      _ = marching_cubes_gpu_pytorch(volume_tensor, threshold)
  torch.cuda.synchronize()
  ```

**GPU Timing (CUDA Synchronized):**

- **Location:** `designB/python/benchmarks.py` lines 55-75
- **Boundary:** `torch.cuda.synchronize()` before and after kernel
- **CAMFM:** A2b_STEADY_STATE (correct timing boundaries)
- **Code:**
  ```python
  torch.cuda.synchronize()
  t_start = time.time()
  vertices, faces = marching_cubes_gpu_pytorch(volume_tensor, threshold)
  torch.cuda.synchronize()
  t_end = time.time()
  gpu_time = t_end - t_start
  ```

**CPU Timing (Baseline):**

- **Location:** `designB/python/benchmarks.py` lines 80-98
- **Implementation:** Direct `time.time()` measurement
- **Code:**
  ```python
  t_start = time.time()
  vertices, faces = marching_cubes_baseline(volume, threshold)
  t_end = time.time()
  cpu_time = t_end - t_start
  ```

**Excluded from Timing:**

- File I/O (volume loading): Before timing starts
- Mesh saving: After timing ends
- Plot generation: After all measurements
- Warmup iterations: Excluded by design

### Performance Characteristics

**43 AFLW2000 Volumes:**

- Average GPU time: 6.6ms per volume
- Average CPU time: 121.2ms per volume
- Speedup: 18.36× (geometric mean)
- Consistency: σ_GPU = 0.1ms, σ_CPU = 5.2ms

**CUDA Kernel Configuration:**

- Block size: 8×8×8 = 512 threads
- Grid size: (24, 24, 25) blocks for 200×192×192 volume
- Total threads: ~7.3 million
- Memory: ~150MB GPU buffers per volume

---

## Design C: GPU-Native Data Pipeline

### Overview

- **Type:** GPU-accelerated data loading + Design B
- **Framework:** PyTorch 2.1.0 + NVIDIA DALI + nvJPEG
- **Dataset Target:** FaceScape (high-resolution, large-scale)
- **Primary Goal:** Minimize CPU↔GPU transfers in data pipeline
- **CAMFM Stages:** DATA.\* stages + all Design B stages

### Implementation Status

- **Current:** Roadmap defined (see `VRN_DesignC_Roadmap.md`)
- **Proposed Entrypoint:** `designC/pipeline/dali_vrn_pipeline.py`
- **Key Innovation:** GPU-resident data pipeline (JPEG decode on GPU)

### Proposed Architecture

```python
# Conceptual entrypoint (not yet implemented)
# File: designC/pipeline/dali_vrn_pipeline.py

import nvidia.dali as dali
import nvidia.dali.fn as fn

@dali.pipeline_def
def vrn_dali_pipeline(data_dir, batch_size):
    # DATA.READ_CPU: Read JPEG bytes from disk
    jpegs, labels = fn.readers.file(file_root=data_dir)

    # DATA.DECODE_GPU_NVJPEG: GPU JPEG decode
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)

    # DATA.RESIZE_GPU: GPU resize to 192×192
    images = fn.resize(images, device='gpu', size=(192, 192))

    # DATA.NORMALIZE_GPU: GPU normalization
    images = fn.normalize(images, device='gpu', mean=[0.485, 0.456, 0.406])

    return images
```

### Configuration Flags (Proposed)

| Flag                   | Default         | Purpose             | CAMFM Stage              |
| ---------------------- | --------------- | ------------------- | ------------------------ |
| `num_threads`          | `4`             | DALI CPU threads    | DATA.READ_CPU            |
| `device_id`            | `0`             | Target GPU          | DATA.DECODE_GPU_NVJPEG   |
| `batch_size`           | `8`             | Pipeline batch size | DATA.DALI_BRIDGE_PYTORCH |
| `prefetch_queue_depth` | `2`             | Pipeline prefetch   | DATA.DECODE_GPU_NVJPEG   |
| `output_dtype`         | `torch.float32` | Tensor output type  | DATA.NORMALIZE_GPU       |

### Expected Speedup

**Hypothetical Performance (vs Design B):**

- Data loading: 3-5× faster (GPU decode + resize)
- Overall pipeline: 20-25× vs Design A
- Memory transfers: 90% reduction (H2D copies)

### Why Not Implemented Yet

1. **Dataset Readiness:** FaceScape preprocessing required
2. **VRN Model Port:** Need PyTorch-native VRN model (currently using Torch7)
3. **Integration Complexity:** DALI → VRN → CUDA MC pipeline needs careful design
4. **Priority:** Focus on Design B completion for thesis deadlines

---

## Design Comparison Matrix

| Feature                   | Design A   | Design A_GPU | Design B               | Design C       |
| ------------------------- | ---------- | ------------ | ---------------------- | -------------- |
| **Framework**             | Torch7     | PyTorch      | PyTorch                | PyTorch + DALI |
| **Device**                | CPU        | GPU          | GPU                    | GPU            |
| **Custom CUDA**           | No         | No           | Yes                    | Yes            |
| **GPU Data Pipeline**     | No         | No           | No                     | Yes            |
| **Speedup (vs A)**        | 1.0×       | ~2-3×        | 18.36×                 | ~20-25×        |
| **CAMFM.A2a**             | ✗          | Partial      | ✓                      | ✓              |
| **CAMFM.A2b**             | ✗          | ✗            | ✓                      | ✓              |
| **CAMFM.A2c**             | ✗          | ✗            | ✓                      | ✓              |
| **CAMFM.A3**              | Partial    | ✗            | ✓                      | ✓              |
| **DATA.\***               | ✗          | ✗            | ✗                      | ✓              |
| **Implementation Status** | Complete   | Conceptual   | Complete               | Roadmap        |
| **Evidence Artifacts**    | 511 meshes | None         | 43 meshes + benchmarks | TBD            |

---

## Thesis Integration Guidelines

### Which Design to Report?

**Primary Focus: Design B**

- Most complete implementation
- Measurable speedup (18.36×)
- Full CAMFM compliance
- Strong evidence bundle

**Secondary: Design A**

- Essential baseline for comparison
- Demonstrates problem scope
- 511 total meshes generated

**Mention Only: Design C**

- Future work section
- Demonstrates understanding of advanced optimization
- Roadmap shows thesis depth

### How to Report Each Design

**Chapter 4.1 (Methodology):**

- Design A: "CPU-only baseline for reproducibility"
- Design B: "CUDA-accelerated with custom marching cubes kernel"
- Design C: "Proposed GPU-native data pipeline extension"

**Chapter 4.2 (Implementation):**

- Design A: Entrypoint + timing methodology
- Design B: Full implementation details + CAMFM mapping
- Design C: Architecture diagram only

**Chapter 5 (Results):**

- Design A vs B comparison
- Speedup analysis (18.36×)
- Quality metrics (Chamfer, F1)

---

## Navigation

- **Pipeline Overview:** See `docs/PIPELINE_OVERVIEW.md`
- **Code Traceability:** See `docs/TRACEABILITY_MATRIX.md`
- **Benchmark Protocol:** See `docs/BENCHMARK_PROTOCOL.md`
- **Detailed Roadmaps:** See `VRN_DesignA_Roadmap.md`, `VRN_DesignB_Roadmap.md`, `VRN_DesignC_Roadmap.md`

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-16  
**Maintainer:** Thesis Author  
**Purpose:** Complete design variant specifications for VRN thesis Chapter 4
