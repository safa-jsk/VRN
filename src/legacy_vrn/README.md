# Legacy VRN Code

This directory contains the original VRN (Volumetric Regression Network) code
from [Jackson et al., 2017](https://aaronsplace.co.uk/papers/jackson2017recon/).

## Contents

| Path | Language | Purpose |
|---|---|---|
| `lua/process.lua` | Torch 7 / Lua | Core VRN inference (requires Docker) |
| `matlab/run.m` | MATLAB | Volume rendering / visualisation |
| `matlab/readvol.m` | MATLAB | Read `.raw` volume files |
| `matlab/rendervol.m` | MATLAB | Render isosurface from volume |
| `python/raw2obj.py` | Python 3 | Convert `.raw` → `.obj` (scikit-image MC) |
| `python/vis.py` | Python 3 | Visualise mesh |
| `shell/run.sh` | Bash | Main entry-point (calls Docker + process.lua) |
| `shell/download.sh` | Bash | Download model weights |
| `third_party/face-alignment/` | Lua | dlib face detection for VRN |

## Running Legacy VRN

```bash
# 1. Pull Docker image
docker pull asjackson/vrn:latest

# 2. Download model weights (one-time)
bash src/legacy_vrn/shell/download.sh

# 3. Run inference
bash src/legacy_vrn/shell/run.sh <input_image>
```

Output: `<image_name>.raw` (200×192×192 uint8 voxel grid)

## Notes

- **Do NOT modify** these files – they are preserved for reproducibility.
- The Lua code requires Torch 7, which only runs inside the Docker container.
- For the GPU-accelerated pipeline, see `src/designB/` (Design B).
