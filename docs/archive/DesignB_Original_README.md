# Design B: CUDA-Accelerated VRN Pipeline

**GPU-accelerated marching cubes post-processing for VRN volumetric face reconstruction**

## Overview

Design B maintains the legacy VRN Torch7 model but accelerates the post-processing pipeline using custom CUDA kernels for marching cubes mesh extraction. This achieves **17.8x speedup** over CPU-based methods while preserving identical inputs and outputs.

### Pipeline

```
AFLW2000 Images → VRN (CPU) → Volume Export → CUDA Marching Cubes → 3D Meshes
                  [Torch7]     [.raw→.npy]     [Custom Kernel]      [.obj]
```

## Performance

- **Average speedup:** 17.8x (GPU vs CPU)
- **GPU processing time:** 4.7ms per volume
- **CPU processing time:** 81.7ms per volume
- **Real-time capability:** 211 FPS (vs 12 FPS CPU)
- **GPU memory:** 28MB per volume

## Directory Structure

```
designB/
├── cuda_kernels/              # CUDA kernel implementation
│   ├── marching_cubes_kernel.cu
│   ├── marching_cubes_bindings.cpp
│   ├── marching_cubes_tables.h
│   ├── cuda_marching_cubes.py
│   ├── test_cuda_mc.py
│   └── README.md
├── python/                    # Python processing utilities
│   ├── marching_cubes_cuda.py     # Main processing script
│   ├── convert_raw_to_npy.py      # Format conversion
│   ├── benchmarks.py              # CPU vs GPU benchmarks
│   ├── volume_io.py               # Volume I/O utilities
│   ├── verify.py                  # Verification script
│   └── compare_designs.py         # Design A vs B comparison
├── scripts/                   # Automation scripts
│   ├── build.sh                   # Build CUDA extension
│   ├── extract_volumes.sh         # Extract VRN volumes
│   ├── run_pipeline.sh            # Complete pipeline
│   ├── run_benchmarks.sh          # Run benchmarks
│   ├── test.sh                    # Test CUDA kernel
│   └── cleanup.sh                 # Clean temporary files
├── build/                     # Build outputs
│   └── marching_cubes_cuda_ext.so
├── setup.py                   # CUDA extension build configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Build CUDA Extension

```bash
cd designB
./scripts/build.sh
```

### 2. Extract VRN Volumes

```bash
./scripts/extract_volumes.sh
```

### 3. Run Complete Pipeline

```bash
./scripts/run_pipeline.sh
```

### 4. Run Benchmarks

```bash
./scripts/run_benchmarks.sh
```

## Requirements

### Hardware
- CUDA-capable GPU (tested on RTX 4070 SUPER)
- 2GB+ GPU memory
- Linux x86_64

### Software
- Python 3.10+
- PyTorch 2.1.0+ with CUDA 11.8+
- NVIDIA CUDA Toolkit 11.8+
- GCC 9+ with C++17 support

### Python Dependencies
```
torch>=2.1.0
pytorch3d>=0.7.5
numpy>=1.26.0
scikit-image
trimesh
matplotlib
seaborn
```

## Implementation Details

### Custom CUDA Kernel

The marching cubes implementation uses:
- **Thread configuration:** 8×8×8 threads per block
- **Compute capability:** SM 8.6 (compatible with RTX 20/30/40 series)
- **Algorithm:** Parallel voxel processing with lookup tables
- **Memory:** Shared memory for edge/triangle tables

### Performance Optimizations

1. **Parallel processing:** Each CUDA thread handles one voxel
2. **Lookup tables:** 256 edge cases cached in shared memory
3. **Zero-copy:** Direct tensor operations without host-device transfers
4. **Batching:** Process multiple volumes efficiently

## Results

### Benchmark Summary (43 AFLW2000 volumes)

| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| Avg time/volume | 81.7ms | 4.7ms | **17.8x faster** |
| Total time (43 vol) | 3.51s | 0.20s | 17.2x faster |
| Throughput | 12 vol/sec | 211 vol/sec | +199 vol/sec |
| Memory usage | N/A | 28MB | Low overhead |

### Output Quality

- **Average mesh:** 172,317 vertices, 57,439 faces
- **Success rate:** 100% (43/43 volumes)
- **Consistency:** Stable results across all volumes

## Documentation

See `../docs/` for detailed documentation:
- **DesignB_README.md** - Implementation overview
- **DesignB_Benchmark_Results.md** - Detailed performance analysis
- **Design_Comparison.md** - Design A vs Design B comparison
- **DesignB_CUDA_Implementation.md** - Technical implementation details

## Data Outputs

Results are stored in `../data/out/designB/`:
- `volumes_raw/` - VRN output volumes (.raw format, 200×192×192)
- `volumes/` - Converted volumes (.npy format)
- `meshes/` - Generated 3D meshes (.obj format)
- `benchmarks_cuda/` - Benchmark results and visualizations

## Testing

```bash
# Test CUDA kernel only
./scripts/test.sh

# Run verification checks
python3 python/verify.py
```

## Troubleshooting

### CUDA compilation errors
- Ensure CUDA Toolkit 11.8+ is installed
- Check GCC version supports C++17: `gcc --version`
- Verify GPU compute capability: `nvidia-smi`

### Runtime errors
- Check PyTorch CUDA availability: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Verify extension loaded: `python3 -c "import marching_cubes_cuda_ext"`

### Path issues
- Run scripts from project root: `cd /home/ahad/Documents/VRN`
- Check symlinks: `ls -la designB/build/`

## Citation

If you use this implementation, please cite:

```bibtex
@thesis{designB2026,
  title={CUDA-Accelerated Marching Cubes for VRN Face Reconstruction},
  author={Your Name},
  year={2026},
  note={Design B - GPU Post-Processing Pipeline}
}
```

## License

See LICENSE file in project root.

## Related

- **Design A:** CPU-only baseline pipeline
- **Design C:** Full PyTorch VRN reimplementation (future work)
