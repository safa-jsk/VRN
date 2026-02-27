# Design A GPU – VRN + GPU-resident Marching Cubes

| Item | Value |
|---|---|
| Framework | Torch 7 (Lua) inside Docker `asjackson/vrn:latest` |
| Post-processing | `process.lua` → `.raw` → **GPU marching cubes** via `src/designB/pipeline.py` |
| CAMFM stage | A2a (GPU residency only, no custom kernel) |

## Quick Start

```bash
# 1. Run VRN inference (Docker)
cd src/legacy_vrn/shell && bash run.sh <image>

# 2. Convert raw → npy
python3 -m src.designB.convert_raw_to_npy --input_dir data/ --output_dir artifacts/volumes/

# 3. Extract mesh using GPU-resident MC (uses PyTorch, no custom CUDA kernel)
python3 -m src.designB.pipeline --input artifacts/volumes/<vol>.npy --output artifacts/meshes/<vol>.obj --threshold 0.5
```

## Notes

This design uses the same GPU marching cubes from Design B but retains the
legacy Torch 7 front-end.  It demonstrates CAMFM A2a (GPU residency) without
the custom CUDA kernel (A2c/A2d).
