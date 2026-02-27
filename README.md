# VRN – Volumetric Regression Network (Model Pipeline – Model 3)

Undergraduate thesis project: GPU-accelerated 3D face reconstruction using
Volumetric Regression Networks with custom CUDA marching cubes.

**Platform:** Ubuntu 24.04.3 · Python 3.10 · PyTorch 2.1.0 · CUDA 11.8

## Quick Start

```bash
# 1. Check environment
bash scripts/env_check.sh

# 2. Build CUDA extensions
bash scripts/build_ext.sh

# 3. Run smoke tests
bash scripts/smoke_test.sh

# 4. Run Design B pipeline (single volume)
python3 -m src.designB.pipeline \
    --input  artifacts/volumes/sample.npy \
    --output artifacts/meshes/sample.obj \
    --threshold 0.5

# 5. Run benchmark (CPU vs GPU)
python3 -m src.designB.benchmark \
    --input_dir  artifacts/volumes/ \
    --output_dir artifacts/benchmarks/
```

## Repository Layout

```
VRN/
├── src/
│   ├── vrn/              # Shared library (config, I/O, perf, metrics, utils)
│   ├── designB/          # Design B: custom CUDA marching cubes pipeline
│   ├── designC/          # Design C: FaceScape extension (skeleton)
│   └── legacy_vrn/       # Original VRN code (Lua/MATLAB/Python, read-only)
├── external/
│   ├── marching_cubes_cuda_ext/   # Custom CUDA kernel + setup.py
│   └── chamfer_ext/               # Chamfer distance CUDA extension
├── DesignA_CPU/          # Wrapper: legacy CPU baseline
├── DesignA_GPU/          # Wrapper: legacy + GPU-resident MC
├── DesignB/              # Wrapper: full GPU pipeline (configs, scripts)
├── DesignC/              # Wrapper: FaceScape extension (configs)
├── experiments/          # Batch runs, comparisons, verification
├── scripts/              # Build, clean, env check, smoke test
├── tests/                # Import + pipeline smoke tests
├── docs/                 # Thesis docs, ops guides, archived reports
├── assets/               # Example images, demo data
├── artifacts/            # Runtime output (git-ignored)
├── requirements.txt      # Python dependencies
└── CITATION.bib
```

## Design Overview

| Design | Description | CAMFM Stages |
|---|---|---|
| **A (CPU)** | Legacy VRN + scikit-image MC on CPU | A0 (baseline) |
| **A (GPU)** | Legacy VRN + GPU-resident MC | A2a |
| **B** | Custom CUDA MC kernel + full pipeline | A2a, A2b, A2c, A3, A5 |
| **C** | FaceScape dataset extension | Same as B (planned) |

## Key CLIs

```bash
# Design B – single inference
python3 -m src.designB.pipeline --input VOL --output MESH --threshold 0.5

# Design B – batch processing
python3 -m src.designB.pipeline --batch --input_dir DIR --output_dir DIR

# Design B – benchmark
python3 -m src.designB.benchmark --input_dir DIR --output_dir DIR --warmup_iters 15

# Design C – FaceScape inference (skeleton)
python3 -m src.designC.infer_facescape --facescape_root DIR --splits_csv CSV

# Design C – FaceScape evaluation (skeleton)
python3 -m src.designC.eval_facescape --pred_dir DIR --gt_dir DIR
```

## CAMFM Methodology

This project follows the CAMFM (CUDA-Accelerated Marching-cubes Face Mesh)
methodology for GPU acceleration.  Key stages:

- **A2a** – GPU residency: data stays on device throughout pipeline
- **A2b** – Steady-state warmup: 15 iterations before timing
- **A2c** – Memory layout: pre-allocated GPU buffers for vertices/triangles
- **A3**  – Metrics: Chamfer distance, F1 score, timing comparisons
- **A5**  – Evidence method: documented in thesis Chapter 4

## Building CUDA Extensions

```bash
bash scripts/build_ext.sh
```

This builds both `marching_cubes_cuda_ext` (SM 8.6) and `chamfer_ext` (SM 8.0)
inside `external/`.

## Documentation

See [docs/index.md](docs/index.md) for the full documentation index.

## Legacy Code

The original VRN Torch 7 code is preserved in `src/legacy_vrn/`.  
See [src/legacy_vrn/README.md](src/legacy_vrn/README.md) for usage instructions.

## License

See [LICENSE](LICENSE).
