# Design B – Custom CUDA Marching Cubes Pipeline

| Item | Value |
|---|---|
| Framework | PyTorch 2.1.0 + CUDA 11.8 + custom CUDA extension |
| Pipeline | `.raw` / `.npy` → **custom CUDA kernel** → `.obj` mesh |
| CAMFM stages | A2a (GPU residency), A2b (steady-state warmup), A2c (memory layout), A3 (metrics), A5 (evidence) |

## Canonical CLIs

```bash
# Single-volume inference
python3 -m src.designB.pipeline \
    --input  artifacts/volumes/sample.npy \
    --output artifacts/meshes/sample.obj \
    --threshold 0.5

# Benchmark (CPU vs GPU, CAMFM-compliant)
python3 -m src.designB.benchmark \
    --input_dir  artifacts/volumes/ \
    --output_dir artifacts/benchmarks/ \
    --warmup_iters 15

# Batch processing
python3 -m src.designB.pipeline \
    --batch \
    --input_dir  artifacts/volumes/ \
    --output_dir artifacts/meshes/ \
    --threshold 0.5
```

## Configuration

Default settings are in `configs/default.yaml`.  Override via CLI flags or
by setting `VRN_CONFIG=path/to/custom.yaml`.

## Directory Map

| Path | Purpose |
|---|---|
| `src/designB/pipeline.py` | Main pipeline (single + batch) |
| `src/designB/benchmark.py` | CPU vs GPU benchmark runner |
| `src/designB/io.py` | Volume I/O and mesh export |
| `src/designB/convert_raw_to_npy.py` | `.raw` → `.npy` converter |
| `external/marching_cubes_cuda_ext/` | Custom CUDA kernel + bindings |
| `configs/default.yaml` | Default YAML config |
