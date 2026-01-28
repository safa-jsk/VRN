# Design A vs Design B Comparison

**Generated:** 2026-01-28 23:18:46  
**Dataset:** AFLW2000-3D subset (43 volumes)  
**GPU:** NVIDIA GeForce RTX 4070 SUPER

## Overview

This report compares the performance of two VRN pipeline designs:

- **Design A:** Original CPU-only pipeline (VRN → scikit-image marching cubes)
- **Design B:** GPU-accelerated pipeline (VRN → Custom CUDA marching cubes)

## Design B Performance Summary

### Marching Cubes Acceleration

| Metric | CPU (scikit-image) | GPU (CUDA kernel) | Improvement |
|--------|-------------------|-------------------|-------------|
| Average time per volume | 84.2 ms | 4.8 ms | **18.04x faster** |
| Total time (43 volumes) | 3.62s | 0.21s | 17.62x faster |
| Min speedup | - | - | 15.33x |
| Max speedup | - | - | 20.18x |
| Throughput | 11.9 volumes/sec | 209.3 volumes/sec | +197.4 vol/sec |

### Real-Time Performance Analysis

| Design | Processing Time | Frame Rate Equivalent | Real-Time Capable? |
|--------|----------------|----------------------|-------------------|
| Design A (CPU) | 84.2 ms/volume | 11.9 FPS | ❌ No (~12 FPS) |
| Design B (GPU) | 4.8 ms/volume | 209.3 FPS | ✅ Yes (~209 FPS) |

**Conclusion:** Design B achieves real-time performance (>30 FPS), making it suitable for interactive applications.

## Pipeline Component Breakdown

### Design A (CPU-only)
```
Input Image (AFLW2000)
    ↓
VRN Forward Pass (Torch7/CPU) [~2-3s per image]
    ↓
Volume Export (.raw format) [~50ms]
    ↓
Marching Cubes (scikit-image/CPU) [84ms]
    ↓
Output Mesh (.obj)
```

**Total pipeline time:** ~2-3 seconds per image (dominated by VRN)

### Design B (CUDA-accelerated)
```
Input Image (AFLW2000)
    ↓
VRN Forward Pass (Torch7/CPU) [~2-3s per image]
    ↓
Volume Export (.npy format) [~50ms]
    ↓
Marching Cubes (Custom CUDA kernel/GPU) [5ms]
    ↓
Output Mesh (.obj)
```

**Total pipeline time:** ~2-3 seconds per image (still dominated by VRN)

## Key Insights

### 1. Marching Cubes Speedup
- **18.0x average speedup** on marching cubes extraction
- GPU processing is **79ms faster** per volume
- Saved **3.41s** across 43 volumes

### 2. Pipeline Impact
- Marching cubes represents **~3.3% of CPU pipeline time**
- After GPU acceleration: **~0.2% of GPU pipeline time**
- **Overall pipeline speedup:** Minimal (~3-4%) because VRN forward pass still dominates

### 3. Use Cases

**Design A (CPU) is best for:**
- One-time dataset processing
- Systems without CUDA-capable GPUs
- Simplicity and minimal dependencies

**Design B (GPU) is best for:**
- Batch processing large datasets
- Real-time or interactive applications
- When VRN volumes are pre-computed
- Research requiring fast iteration

### 4. Resource Utilization

| Resource | Design A | Design B |
|----------|----------|----------|
| GPU Memory | 0 MB | ~28 MB |
| CPU Utilization | High (100% single core) | Low (data transfer only) |
| Scalability | Limited (single-threaded) | Excellent (parallel) |

## Recommendations

### When to Use Design B

1. **Pre-computed volumes:** If VRN volumes are already available, Design B provides **18x faster** mesh extraction
2. **Batch processing:** Processing 43 volumes saves **3.4s** (94% time reduction)
3. **Interactive applications:** 209 FPS throughput enables real-time visualization
4. **Research workflows:** Fast iteration for parameter tuning and experimentation

### When to Use Design A

1. **Single image processing:** Speedup benefit is negligible for 1-2 images
2. **No GPU available:** CPU-only systems or cloud instances without CUDA
3. **Production stability:** Fewer dependencies, simpler deployment

## Implementation Comparison

| Aspect | Design A | Design B |
|--------|----------|----------|
| Dependencies | Python, scikit-image | Python, PyTorch, CUDA 11.8+, Custom kernel |
| Complexity | Low | Medium (CUDA kernel) |
| Setup Time | <5 minutes | ~30 minutes (build CUDA extension) |
| Code Maintenance | Minimal | Moderate (kernel updates) |
| Portability | High (pure Python) | Medium (CUDA-capable GPU required) |

## Performance Metrics Detail

### Top 5 Fastest Speedups

1. **image00036.npy:** 20.18x (88.6ms → 4.6ms)
2. **image00019.npy:** 19.87x (86.3ms → 4.5ms)
3. **image00035.npy:** 19.30x (84.2ms → 4.5ms)
4. **image00074.npy:** 19.22x (85.5ms → 4.6ms)
5. **image00004.npy:** 18.97x (85.3ms → 4.6ms)

### Bottom 5 Speedups (Still Significant)

1. **image00002.jpg.crop.npy:** 15.33x (94.6ms → 7.3ms)
2. **image00063.npy:** 16.00x (82.5ms → 5.4ms)
3. **image00044.npy:** 17.08x (80.4ms → 4.8ms)
4. **image00051.npy:** 17.39x (82.2ms → 4.6ms)
5. **image00056.npy:** 17.43x (79.9ms → 4.6ms)

## Conclusion

Design B's custom CUDA marching cubes kernel provides **18.0x average speedup** over Design A's CPU implementation. While this represents only a **~3.2% improvement** in total pipeline time (due to VRN bottleneck), it enables:

✅ **Real-time mesh extraction** (209 FPS vs 12 FPS)  
✅ **Efficient batch processing** (3.4s saved on 43 volumes)  
✅ **Low GPU memory overhead** (28MB allocation)  
✅ **Consistent performance** (15-19x speedup across all volumes)  

**Recommendation:** Deploy Design B for batch processing and interactive applications where VRN volumes are pre-computed or when fast marching cubes extraction is critical.

---

*Generated from benchmarks run on 2026-01-28 23:18:46*  
*Dataset: AFLW2000-3D (43 volumes, 200×192×192 voxels each)*  
*Hardware: NVIDIA GeForce RTX 4070 SUPER*
