# Batch Processing Summary - 1000 Images

## Execution Details

### Command

```bash
./scripts/batch_process_aflw2000.sh docs/aflw2000_subset_1000.txt
```

### Subset Configuration

- **Total AFLW2000 images available**: 2000
- **Subset size**: 1000 images (first half, sorted alphabetically)
- **Subset file**: `docs/aflw2000_subset_1000.txt`
- **Demo run**: 50 images for testing

### Subset File Contents

The subset file contains the first 1000 images from the AFLW2000 dataset:

- Format: One filename per line (e.g., `image00002.jpg`)
- Location: `/home/safa-jsk/Documents/VRN/docs/aflw2000_subset_1000.txt`
- Verification: `wc -l docs/aflw2000_subset_1000.txt` → **1000 lines**

## Processing Pipeline

### Design A - VRN Pipeline

Each image undergoes:

1. **Input**: Copy to Docker volume (`data/in/aflw2000/`)
2. **Face Detection**: Torch7-based landmark detection
3. **Face Alignment**: 3D face model fitting
4. **Volumetric Reconstruction**: VRN volumetric neural network
5. **Mesh Extraction**: Marching cubes → OBJ output
6. **Output**: Mesh saved to `data/out/designA/imageXXXXX.jpg.obj`

### Performance Characteristics

- **Per-image time**: ~20-30 seconds (Docker + Torch7 inference)
- **Total estimated time (1000 images)**: 5-8 hours
- **Per-image throughput**: ~40 images/hour at ~150 seconds/image

### Demonstration Run

- **Demo images**: 50 (subset of 1000)
- **Demo estimated time**: ~20-25 minutes
- **Status**: Running in background (PID: 354361)

## Output Locations

### Per-Image Results

- **Meshes**: `data/out/designA/imageXXXXX.jpg.obj` (OBJ format)
- **Volume files**: `data/out/designA/imageXXXXX.jpg.npy` (volumetric grid)

### Batch Results

- **Batch log**: `data/out/designA/batch_process.log` (per-image processing details)
- **Timing log**: `data/out/designA/time.log` (elapsed seconds per image)
- **Metrics report**: `data/out/designA/metrics/mesh_metrics.csv` (Chamfer, F1 scores)
- **Analysis report**: `data/out/designA/analysis_report.md` (summary statistics)

## Batch Script Enhancements

### New Features (Subset Support)

```bash
# Usage: ./batch_process_aflw2000.sh [subset_file]
# If no subset_file provided: processes all images in INPUT_DIR
```

### Modifications Made

1. **Subset file parameter**: Optional first argument to specify image list
2. **Flexible image collection**: Pre-collects images array (fixes premature exit bug)
3. **Per-sample timing**: Each image logs elapsed time to `time.log`
4. **Metrics integration**: Automatically runs `analyze_results.sh` upon completion

### Timing Log Format

```
# Timing Log - Format: seconds KB filename
15 150000 image00002.jpg
22 142000 image00004.jpg
20 148000 image00006.jpg
... (per-image entries)
TOTAL_BATCH_TIME: 1234 sec (aggregated)
```

## Next Steps

### Monitor Progress

```bash
# View current processing
tail -f data/out/designA/batch_process.log

# Check timing log
tail -f data/out/designA/time.log

# Monitor background process
ps aux | grep batch_process_aflw2000
```

### After Completion

1. **Review metrics**: `data/out/designA/metrics/mesh_metrics.csv`
2. **Check statistics**: `data/out/designA/analysis_report.md`
3. **Verify outputs**: Count meshes: `find data/out/designA -name "*.obj" | wc -l`
4. **Compare timings**: Aggregate from `time.log`

### For Full 1000-Image Run

To process all 1000 images:

```bash
./scripts/batch_process_aflw2000.sh docs/aflw2000_subset_1000.txt
# Estimated completion time: 5-8 hours
```

## Error Handling

### Common Issues

1. **"No images found" (FIXED)**: Script now pre-collects image array, validates once before loop
2. **Docker timeout**: Docker container slow on CPU inference (expected behavior for legacy Design A)
3. **Missing output files**: Check `batch_process.log` for per-image errors

### Debugging

- Check batch log: `tail -100 data/out/designA/batch_process.log`
- Search for errors: `grep -i "error\|failed" data/out/designA/batch_process.log`
- Verify Docker: `docker ps` and `docker logs <container_id>`

## Architecture Notes

### Design A (CPU-based)

- Docker image: `asjackson/vrn:latest`
- Framework: Torch7 (legacy)
- Processing: CPU-only (no CUDA)
- Speed: Slower due to CPU inference

### Design B (GPU-accelerated)

- PyTorch 2.1.0 + CUDA 11.8
- Custom CUDA kernels (marching cubes)
- GPU inference (orders of magnitude faster)
- Can process 1000 images in ~10-30 minutes

## Metrics Computation

### Chamfer Distance

- **GPU method**: CUDA extension (prebuilt in `chamfer/build/`)
- **CPU fallback**: scipy.spatial.cKDTree
- **Computation**: Point-cloud distance between predicted and ground truth meshes

### F1 Metrics

- **F1_tau**: F1 score at threshold τ (default: 1.0)
- **F1_2tau**: F1 score at 2τ threshold
- **Precision/Recall**: Per-threshold variants

### CSV Output Columns

- `mesh_name`: Image filename
- `pred_mesh`: Path to predicted mesh
- `gt_mesh`: Path to ground truth mesh
- `chamfer_mean`: Mean Chamfer distance
- `chamfer_std`: Std dev of Chamfer distance
- `f1_tau`: F1 score at threshold τ
- `f1_2tau`: F1 score at 2τ
- `precision_tau`, `recall_tau`: Per-threshold metrics
- ... (additional precision/recall for 2τ variants)

---

**Created**: February 1, 2026  
**Dataset**: AFLW2000-3D  
**Processing Method**: VRN Design A (CPU)  
**Status**: Batch processing running on 50-image demo subset
