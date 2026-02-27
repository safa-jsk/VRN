# Design A – CPU (Legacy VRN + scikit-image Marching Cubes)

| Item | Value |
|---|---|
| Framework | Torch 7 (Lua) inside Docker `asjackson/vrn:latest` |
| Post-processing | `process.lua` → `.raw` → `raw2obj.py` (scikit-image MC on CPU) |
| CAMFM stage | A0 (baseline, no GPU acceleration) |

## Quick Start

```bash
# Requires: Docker + Lua model weights (download.sh)
cd src/legacy_vrn/shell && bash run.sh <image>
python3 src/legacy_vrn/python/raw2obj.py --input <vol>.raw --output mesh.obj
```

## Directory Map

Source code lives in `src/legacy_vrn/`.  See [src/legacy_vrn/README.md](../src/legacy_vrn/README.md).
