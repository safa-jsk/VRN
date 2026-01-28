# Custom CUDA Marching Cubes Kernel

GPU-accelerated marching cubes implementation for PyTorch using custom CUDA kernels.

## Performance

- **5.26x faster** than CPU (scikit-image) on test volumes
- **~4ms** processing time for 200×192×192 VRN volumes
- Runs on NVIDIA GPUs (SM 6.0+, tested on RTX 4070 SUPER)

## Files

- `marching_cubes_kernel.cu` - CUDA kernel implementation
- `marching_cubes_bindings.cpp` - PyTorch C++ bindings
- `marching_cubes_tables.h` - Lookup tables (Paul Bourke)
- `cuda_marching_cubes.py` - Python wrapper
- `test_cuda_mc.py` - Test script

## Quick Start

### 1. Build Extension
```bash
cd /path/to/VRN
source vrn_env/bin/activate
python3 setup.py build_ext --inplace
```

### 2. Test
```bash
python3 cuda_kernels/test_cuda_mc.py
```

Expected output:
```
✓ GPU: NVIDIA GeForce RTX 4070 SUPER
CPU Time: 0.0290s | 19,008 vertices
GPU Time: 0.0055s | 57,030 vertices
SPEEDUP: 5.26x ✓
```

### 3. Use in Code
```python
import torch
import marching_cubes_cuda_ext

# Load volume as torch tensor on GPU
volume = torch.randn(200, 192, 192).cuda()

# Allocate outputs
max_verts = volume.numel() * 15
max_tris = volume.numel() * 5
vertices = torch.zeros((max_verts, 3), dtype=torch.float32, device='cuda')
triangles = torch.zeros((max_tris, 3), dtype=torch.int32, device='cuda')
num_verts = torch.zeros(1, dtype=torch.int32, device='cuda')
num_tris = torch.zeros(1, dtype=torch.int32, device='cuda')

# Run marching cubes
marching_cubes_cuda_ext.marching_cubes_forward(
    volume, vertices, triangles, num_verts, num_tris,
    0.5,  # isolevel
    192, 192, 200,  # dimX, dimY, dimZ
    max_verts, max_tris
)

# Get results
nv = num_verts.item()
nt = num_tris.item()
final_verts = vertices[:nv, :].cpu().numpy()
final_faces = triangles[:nt, :].cpu().numpy()
```

## Requirements

- **Python:** 3.10+
- **PyTorch:** 2.1.0+ with CUDA
- **CUDA Toolkit:** 11.8+ (nvcc)
- **GPU:** NVIDIA with compute capability 6.0+

## Architecture

```
marching_cubes_kernel.cu (CUDA)
    ↓ (nvcc compile)
marching_cubes_bindings.cpp (PyBind11)
    ↓ (link)
marching_cubes_cuda_ext.so (Python module)
    ↓ (import)
cuda_marching_cubes.py (High-level API)
```

## Technical Details

### CUDA Kernel Configuration
- **Block Size:** 8×8×8 threads (512 per block)
- **Grid Size:** Dynamically calculated based on volume dimensions
- **Memory:** Dynamic allocation with atomic counters
- **Stream:** Default CUDA stream (0)

### Compiler Flags
```bash
-O3                      # Max optimization
--use_fast_math          # Fast GPU math
-arch=sm_86              # Target architecture
-std=c++17               # C++17 features
--expt-relaxed-constexpr # Relaxed constexpr
```

## Troubleshooting

### Build Fails: `sm_89 not defined`
**Solution:** Your CUDA toolkit doesn't support SM 8.9. The code automatically uses SM 8.6 (max for CUDA 11.8), which runs on newer GPUs via forward compatibility.

### Import Error: `cannot import marching_cubes_cuda_ext`
**Solution:** Run `python3 setup.py build_ext --inplace` from VRN root directory.

### Runtime Error: `CUDA not available`
**Solution:** 
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Performance Tips

1. **Batch Processing:** Reuse allocated buffers across multiple volumes
2. **Stream Parallelism:** Use multiple CUDA streams for concurrent execution
3. **Memory Pool:** Use PyTorch's memory pool for faster allocation
4. **Precision:** Use FP16 if your GPU supports it (RTX series)

## Future Enhancements

- [ ] Full Paul Bourke lookup tables (256 cube configurations)
- [ ] GPU-side normal computation
- [ ] Shared memory optimization
- [ ] Multi-stream batch processing
- [ ] FP16/TF32 support

## License

Part of VRN project. Custom CUDA kernel based on public domain marching cubes algorithm (Lorensen & Cline, 1987) and lookup tables (Paul Bourke).

## Contact

For issues or questions about this CUDA implementation, see main VRN repository.
