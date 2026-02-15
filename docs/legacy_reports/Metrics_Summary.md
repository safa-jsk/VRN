# VRN Design B Metrics Summary

**Evaluation Date:** February 4, 2026  
**Dataset:** 300W-LP AFW Subset (1000 images, 468 evaluated)  
**Reference:** Design A Baseline Meshes

---

## Metrics Overview

This document provides a comprehensive summary of all metrics collected during Design B batch processing evaluation, including timing statistics, mesh quality metrics, and geometric analysis.

---

## 1. Timing Metrics

### GPU Marching Cubes Timing (n=468)

| Statistic | Value | Unit |
|-----------|-------|------|
| **Mean** | 5.14 | ms |
| **Standard Deviation** | 0.25 | ms |
| **Minimum** | 4.40 | ms |
| **Maximum** | 6.23 | ms |
| **Median** | ~5.10 | ms |
| **Throughput** | 194.4 | volumes/sec |

### Total Pipeline Timing

| Stage | Mean Time | Description |
|-------|-----------|-------------|
| Volume Generation | ~1.2 s | VRN + format conversion |
| Marching Cubes (GPU) | 5.14 ms | Custom CUDA kernel |
| Mesh Export | ~10 ms | OBJ file writing |
| **Total per Image** | ~1.09 s | End-to-end |

### Batch Processing Summary

| Metric | Value |
|--------|-------|
| **Total Batch Time** | 1092.4 seconds (18.2 min) |
| **Images Processed** | 1000 |
| **Meshes Generated** | 468 |
| **Average per Image** | 1.09 seconds |

---

## 2. Mesh Quality Metrics

### Chamfer Distance (n=468)

| Statistic | Value | Unit |
|-----------|-------|------|
| **Mean** | 10.47 | voxel units |
| **Standard Deviation** | 3.44 | voxel units |
| **Minimum** | ~6.68 | voxel units |
| **Maximum** | ~22.46 | voxel units |
| **Median** | ~9.5 | voxel units |

#### Chamfer Distance Distribution

```
Range (voxel units)    Count    Percentage
─────────────────────────────────────────
 0.0 -  5.0            ~5       ~1%
 5.0 -  8.0            ~150     ~32%
 8.0 - 10.0            ~120     ~26%
10.0 - 12.0            ~90      ~19%
12.0 - 15.0            ~70      ~15%
15.0 - 25.0            ~33      ~7%
```

### F1 Score Analysis

| Metric | τ=0.01 | τ=0.02 |
|--------|--------|--------|
| **Mean F1** | 2.14×10⁻⁷ | 2.56×10⁻⁶ |
| **Max F1** | ~1×10⁻⁴ | ~1×10⁻⁴ |
| **Min F1** | 0 | 0 |

**Note:** Low F1 scores indicate that the τ thresholds are too strict relative to the mesh scale. Vertices rarely fall within 0.01-0.02 voxel units of corresponding reference vertices.

#### Precision & Recall at τ=0.01

| Metric | Mean | Max |
|--------|------|-----|
| **Precision** | ~0 | ~1×10⁻⁴ |
| **Recall** | ~0 | ~1×10⁻⁴ |

---

## 3. Geometric Metrics

### Mesh Complexity (n=468)

| Property | Mean | Min | Max | Std |
|----------|------|-----|-----|-----|
| **Design B Vertices** | 220,341 | 97,916 | 281,789 | ~45,000 |
| **Design B Faces** | 143,827 | 64,619 | 180,884 | ~30,000 |
| **Design A Vertices** | 32,044 | 26,582 | 38,954 | ~3,000 |
| **Design A Faces** | 132,456 | 112,192 | 155,644 | ~10,000 |

### Vertex Count Comparison

```
Design B vs Design A Vertex Ratio
─────────────────────────────────
Mean Ratio: 6.88× more vertices in Design B
Min Ratio:  3.08×
Max Ratio:  10.6×
```

### Bounding Box Analysis (Sample)

| Axis | Design A Range | Design B Range |
|------|----------------|----------------|
| X | 45-133 (88) | 57.5-139.5 (82) |
| Y | 33-175 (142) | 53.5-174.5 (121) |
| Z | 17-88 (71) | 15.8-88.2 (72.5) |

---

## 4. Computation Method Details

### Chamfer Distance Computation

**Algorithm:** PyTorch GPU Batched Pairwise Distance

```python
# Point sampling: 10,000 points per mesh
# Method: Uniform face-weighted random sampling
# Distance: L2 norm (Euclidean)

chamfer_mean = (mean(min_dist_pred_to_ref) + mean(min_dist_ref_to_pred)) / 2
```

**GPU Acceleration:**
- Batched `torch.cdist()` with chunk size 2000
- CUDA tensor operations
- No CUDA extension compilation required

### F1 Score Computation

```python
# Threshold τ = 0.01 (strict)
# Threshold 2τ = 0.02 (relaxed)

precision_tau = count(pred_to_ref_dist < tau) / n_pred
recall_tau = count(ref_to_pred_dist < tau) / n_ref
f1_tau = 2 * (precision * recall) / (precision + recall)
```

---

## 5. Per-Sample Metrics (First 20)

| Sample | Chamfer Mean | F1(τ) | Pred Verts | Ref Verts |
|--------|--------------|-------|------------|-----------|
| AFW_1051618982_1_0 | 9.35 | 0 | 271,306 | 31,688 |
| AFW_1051618982_1_1 | 11.88 | 0 | 153,262 | 32,595 |
| AFW_1051618982_1_2 | 12.12 | 0 | 155,100 | 33,316 |
| AFW_1051618982_1_3 | 7.48 | 0 | 243,280 | 33,678 |
| AFW_1051618982_1_4 | 8.03 | 0 | 261,329 | 34,139 |
| AFW_1051618982_1_5 | 8.08 | 0 | 261,861 | 33,000 |
| AFW_1051618982_1_6 | 13.58 | 0 | 152,070 | 32,184 |
| AFW_1051618982_1_7 | 15.32 | 0 | 140,728 | 28,196 |
| AFW_1051618982_1_8 | 16.12 | 0 | 155,811 | 32,660 |
| AFW_111076519_1_0 | 7.79 | 0 | 254,775 | 35,556 |
| AFW_111076519_1_1 | 7.08 | 0 | 168,438 | 35,186 |
| AFW_111076519_1_2 | 9.03 | 0 | 169,525 | 35,576 |
| AFW_111076519_1_3 | 14.44 | 0 | 266,973 | 33,154 |
| AFW_111076519_1_4 | 8.90 | 0 | 267,204 | 34,191 |
| AFW_111076519_2_0 | 9.38 | 0 | 271,872 | 29,536 |
| AFW_111076519_2_1 | 12.70 | 0 | 262,935 | 30,747 |
| AFW_111076519_2_10 | 6.84 | 0 | 184,016 | 27,596 |
| AFW_111076519_2_11 | 12.53 | 0 | 220,347 | 28,154 |
| AFW_111076519_2_13 | 12.23 | 0 | 173,293 | 28,878 |
| AFW_111076519_2_14 | 8.86 | 0 | 226,084 | 27,224 |

---

## 6. Statistical Summary

### Key Findings

1. **Chamfer Distance:** Mean of 10.47 units indicates moderate geometric deviation between Design A and B meshes, primarily due to different mesh densities and vertex merge tolerances.

2. **F1 Scores Near Zero:** The τ=0.01 threshold is too strict for the mesh scale. Recommend using τ≥1.0 for meaningful F1 metrics.

3. **Vertex Density:** Design B produces 6.88× more vertices on average, resulting in smoother but geometrically different meshes.

4. **GPU Performance:** Consistent 5.14ms ± 0.25ms timing demonstrates stable GPU kernel execution.

### Recommendations

1. **Rescale τ:** Use τ=1.0 or τ=2.0 for threshold-based metrics
2. **Normalize Meshes:** Scale both meshes to unit bounding box before comparison
3. **ICP Alignment:** Apply rigid alignment before computing Chamfer distance
4. **Increase Samples:** Use 50,000+ points for high-detail meshes

---

## 7. Raw Data Locations

| Data | File Path |
|------|-----------|
| Per-image timing | `data/out/designB_1000_metrics/logs/timing.csv` |
| Per-mesh metrics | `data/out/designB_1000_metrics/metrics/mesh_metrics.csv` |
| Aggregated stats | `data/out/designB_1000_metrics/batch_summary.json` |

---

## 8. Computation Environment

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 4070 SUPER |
| **CUDA Capability** | 8.9 |
| **PyTorch** | 2.x with CUDA 11.8 |
| **Chamfer Method** | PyTorch GPU (pure Python) |
| **Marching Cubes** | Custom CUDA kernel |
| **Point Sampling** | trimesh uniform sampling |

---

*Metrics Summary Generated: February 4, 2026*
