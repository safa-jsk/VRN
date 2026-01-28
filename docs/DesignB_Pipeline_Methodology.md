# Design B: Pipeline & Methodology

**GPU-Accelerated Post-Processing for VRN Face Reconstruction**

---

## Overview

Design B implements GPU-accelerated post-processing for the VRN pipeline while keeping the legacy Torch7 VRN model unchanged. This approach achieves measurable CUDA acceleration without requiring modification of the legacy neural network implementation.

## Core Methodology

**Two-Stage Pipeline:**
- **Stage 1:** VRN Volume Extraction (CPU - Legacy Torch7)
- **Stage 2:** CUDA Marching Cubes (GPU - Custom Kernel)

---

## Pipeline Architecture

```
Input: AFLW2000 Face Images (.jpg)
  ↓
┌─────────────────────────────────────────────────┐
│ STAGE 1: VRN Volume Regression (CPU)           │
│ - Docker: asjackson/vrn:latest                 │
│ - Engine: Torch7 (Legacy)                      │
│ - Model: vrn-unguided.t7                       │
│ - Script: process.lua (modified)               │
│ - Output: 200×192×192 voxel volumes (.raw)     │
└─────────────────────────────────────────────────┘
  ↓
Format Conversion
  - Input: .raw (uint8)
  - Output: .npy (float32)
  - Script: convert_raw_to_npy.py
  ↓
┌─────────────────────────────────────────────────┐
│ STAGE 2: GPU Marching Cubes (CUDA)             │
│ - Implementation: Custom CUDA kernel           │
│ - Language: CUDA C++ + PyTorch bindings        │
│ - Thread Config: 8×8×8 blocks                  │
│ - Compute: SM 8.6 (RTX compatible)             │
│ - Processing: Parallel voxel isosurface        │
│ - Output: 3D meshes (.obj format)              │
└─────────────────────────────────────────────────┘
  ↓
Output: 3D Face Meshes (.obj)
```

---

## Implementation Details

### 1. VRN Volume Extraction

**Location:** `designB/scripts/extract_volumes.sh`

**Process:**
- Mount AFLW2000 images into VRN Docker container
- Run modified `process.lua` script
- Export volumetric predictions as `.raw` files
- Generate 200×192×192 uint8 voxel grids per face

**Key Modification:**
- Added volume export capability to `process.lua`
- Bypassed default mesh generation
- Direct `.raw` volume output for downstream processing

### 2. Format Conversion

**Location:** `designB/python/convert_raw_to_npy.py`

**Process:**
1. Read VRN `.raw` binary format (200×192×192 uint8)
2. Convert to NumPy `.npy` format (float32)
3. Normalize voxel values to [0, 1] range
4. Enable Python/PyTorch processing

### 3. Custom CUDA Marching Cubes

**Location:** `designB/cuda_kernels/`

**Components:**

#### a) CUDA Kernel (`marching_cubes_kernel.cu`)
- Parallel voxel processing
- Lookup table-based edge computation
- Triangle generation per voxel
- Optimized memory access patterns

#### b) PyTorch Bindings (`marching_cubes_bindings.cpp`)
- C++17 pybind11 interface
- Tensor memory management
- CUDA stream handling
- Python-friendly API

#### c) Lookup Tables (`marching_cubes_tables.h`)
- 256 edge table entries
- Triangle configuration table
- Shared memory optimization

#### d) Python Wrapper (`cuda_marching_cubes.py`)
- High-level API
- Device management
- Error handling
- CPU fallback support

### 4. Build System

**Location:** `designB/setup.py`

**Configuration:**
- `CUDAExtension` for PyTorch
- C++17 standard
- SM 8.6 compute capability
- Ninja build backend
- Automatic dependency detection

---

## Methodology Rationale

### Design Decision 1: Keep VRN Unchanged

**Reason:** Legacy Torch7 + CUDA 7.5 incompatible with modern GPUs

**Solution:** Run VRN on CPU, accelerate post-processing only

**Justification:** Maintains reproducibility while demonstrating CUDA acceleration

### Design Decision 2: Custom CUDA Kernel

**Reason:** Demonstrate CUDA expertise, control over optimization

**Advantage:** 18.36x speedup vs CPU baseline

**Alternative Considered:** Use existing libraries (PyTorch3D, Kaolin)

**Why Custom:** Greater control, learning opportunity, optimization potential

### Design Decision 3: Two-Stage Pipeline

**Reason:** Separation of concerns, modularity

**Benefits:**
- Can reuse VRN volumes for multiple experiments
- GPU processing independent of VRN updates
- Clear performance attribution
- Easier debugging and validation

### Design Decision 4: Volume Intermediate Format

**Reason:** Flexibility, debugging, analysis

**Format:** `.npy` (NumPy native, Python-friendly)

**Size:** 1.2GB for 43 volumes

**Advantage:** Easy inspection, format conversion, reprocessing

---

## Algorithm Implementation

### Marching Cubes Algorithm (GPU)

**Input:** 3D voxel grid V[x,y,z], threshold T

**Process (per voxel, in parallel):**
1. Sample 8 corner values
2. Compute cube index (0-255) based on threshold
3. Lookup edge table → determine active edges
4. Interpolate vertex positions on active edges
5. Lookup triangle table → generate triangles
6. Write vertices and faces to output buffers

**CUDA Optimization:**
- **Thread per voxel:** (200-1) × (192-1) × (192-1) threads
- **Block size:** 8×8×8 = 512 threads
- **Grid:** Automatic calculation based on volume dimensions
- **Memory:** Coalesced access, shared lookup tables
- **Synchronization:** Minimal, embarrassingly parallel workload

---

## Performance Characteristics

### Measured Performance (RTX 4070 SUPER)

| Metric | CPU (scikit-image) | GPU (Custom CUDA) | Speedup |
|--------|-------------------|-------------------|---------|
| Average time/volume | 84.2ms | 4.6ms | **18.36x** |
| Best case | - | - | 19.58x |
| Worst case | - | - | 16.51x |
| Std deviation | ±3.1ms | ±0.2ms | Consistent |

### Resource Utilization

| Resource | CPU | GPU |
|----------|-----|-----|
| Memory | N/A | 28MB per volume |
| VRAM Reserved | N/A | 1720MB total |
| CPU Usage | 100% (single core) | <5% (data transfer only) |
| GPU Utilization | N/A | Brief bursts (~10ms) |

### Throughput

| Metric | Value |
|--------|-------|
| CPU throughput | 12 volumes/second |
| GPU throughput | 217 volumes/second |
| Real-time equivalent | 217 FPS |

---

## Validation Methodology

### 1. Correctness Validation

- Compare vertex counts with CPU baseline
- Visual inspection of meshes
- Mesh topology verification
- **Success rate:** 100% (43/43 volumes)

### 2. Performance Validation

- 3 runs per volume (minimize variance)
- Best-of-N timing approach
- GPU synchronization before timing
- Warm-up runs to eliminate cold-start effects

### 3. Consistency Checks

- Identical input volumes as Design A
- Same AFLW2000 subset (43 images)
- Deterministic outputs
- Reproducible builds

---

## Dataset

**Source:** AFLW2000-3D

**Subset:** 43 images (documented in `docs/aflw2000_subset.txt`)

**Selection Criteria:** Successfully processed by Design A

**Format:** `.jpg` face crops

**Volume Dimensions:** 200×192×192 voxels per face

---

## Outputs

**Mesh Files:** 43 `.obj` files

**Average Mesh:**
- Vertices: 63,571
- Faces: ~127,000

**Format:** Wavefront OBJ (industry standard)

**Location:** `data/out/designB/meshes/`

---

## Reproducibility

### Build and Execution Commands

```bash
# Build CUDA extension
./designB/scripts/build.sh

# Extract VRN volumes
./designB/scripts/extract_volumes.sh

# Run complete pipeline
./designB/scripts/run_pipeline.sh

# Run benchmarks
./designB/scripts/run_benchmarks.sh
```

### Dependencies

**System Requirements:**
- CUDA-capable GPU (SM 5.0+, tested on RTX 4070 SUPER)
- NVIDIA Driver 470+
- Docker (for VRN container)

**Software Stack:**
- Python 3.10.12
- PyTorch 2.1.0+cu118
- PyTorch3D 0.7.5
- CUDA Toolkit 11.8
- GCC 9+ (C++17 support required)
- NumPy 1.26.4
- scikit-image 0.22+
- trimesh 4.0+

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/DesignB_CUDA_Implementation.md` | Technical implementation details |
| `docs/DesignB_Benchmark_Results.md` | Detailed performance analysis |
| `docs/Design_Comparison.md` | Design A vs Design B comparison |
| `designB/README.md` | Quick start guide |

---

## Scientific Contribution

Design B demonstrates that:

✅ **Legacy models can be accelerated via modern CUDA** without retraining or architectural changes

✅ **Modular post-processing achieves significant speedup** (18.36x) through targeted optimization

✅ **Two-stage pipelines maintain reproducibility** while enabling GPU acceleration

✅ **Custom kernels can outperform library implementations** through domain-specific optimization

---

## Limitations

### Performance Limitations

- **VRN inference remains CPU-bound** (~2-3s per image)
- **Total pipeline speedup only ~3%** (VRN forward pass dominates)
- **GPU acceleration most beneficial** for batch processing pre-computed volumes

### Hardware Requirements

- Requires CUDA-capable GPU (SM 5.0+ minimum)
- Tested on RTX 4070 SUPER (SM 8.9)
- Compiled for SM 8.6 (forward compatible)

### Use Case Constraints

- Best for batch processing scenarios
- Limited benefit for single-image inference
- VRN Docker container dependency

---

## Future Work (Design C)

**Planned Enhancements:**
- PyTorch VRN reimplementation for end-to-end GPU acceleration
- Training pipeline on 300W-LP dataset
- Real-time inference (<100ms total pipeline time)
- Reduced dependency on legacy Docker container

**Expected Impact:**
- Full GPU utilization throughout pipeline
- Sub-second inference for real-time applications
- Simplified deployment without Docker dependency

---

## References

**Original VRN Paper:**
- Jackson, A. S., et al. "Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression." ICCV 2017.

**Marching Cubes Algorithm:**
- Lorensen, W. E., & Cline, H. E. "Marching cubes: A high resolution 3D surface construction algorithm." ACM SIGGRAPH 1987.

**CUDA Programming:**
- NVIDIA CUDA C++ Programming Guide
- PyTorch C++ Extension Documentation

---

*Document generated: January 29, 2026*  
*Implementation: Design B - CUDA-Accelerated VRN Pipeline*  
*Hardware: NVIDIA GeForce RTX 4070 SUPER*  
*Dataset: AFLW2000-3D (43 image subset)*
