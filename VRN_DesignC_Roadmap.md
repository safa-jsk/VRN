# VRN — Design C Roadmap (CUDA + Modernization)

**Goal (Design C):** Deliver a **modern, maintainable, CUDA-accelerated** 3D face reconstruction pipeline inspired by VRN, with an engineering focus on:
- reproducibility (Docker/conda, pinned versions),
- GPU acceleration (end-to-end inference on RTX-class GPUs),
- and improved developer ergonomics (Python 3.x, modern PyTorch).

Design C is the “full” upgrade path beyond Design A (legacy CPU) and Design B (partial CUDA).

---

## 1) Scope and success criteria

### What Design C includes
- Modernize the implementation to a **current framework (PyTorch)** and **current CUDA** stack.
- Enable **end-to-end GPU inference** (model + post-processing).
- Provide reproducible environment(s):
  - CUDA-enabled Docker image **or**
  - conda environment + documented setup
- Run experiments on:
  - 300W-LP (training / fine-tuning) — as feasible
  - AFLW2000-3D (evaluation subset)

### What Design C may include (optional)
- Faster/different face alignment (e.g., replacing legacy dlib-based detector if needed)
- Better mesh texturing/colouring pipeline
- Quantitative evaluation metrics (NME, depth error, mesh distance)

### Success criteria
- ✅ Pipeline runs on RTX 2050 (or similar) with CUDA acceleration.
- ✅ Produces meshes comparable to Design A outputs on the same test subset.
- ✅ Has a reproducible build (Dockerfile or environment.yml).
- ✅ Provides performance gains vs Design A baseline and/or improved robustness.

---

## 2) Design philosophy (what you’ll justify in Chapter 4)

### Why modernization is needed
- Legacy VRN stack depends on very old CUDA/cuDNN and Torch7.
- Modern CUDA GPUs require current toolchains to run efficiently and reliably.
- Modernization improves:
  - maintainability
  - reproducibility
  - compatibility with current drivers
  - profiling and optimization workflows

### Design C objective framing
- Treat Design C as a **system re-design** while preserving the original methodological intent:
  - single-image → volumetric representation → mesh reconstruction

---

## 3) Architecture overview (Design C)

### Proposed modern pipeline
1) Input image
2) Face detection + alignment / crop
3) Neural inference (PyTorch, CUDA)
4) Volume (or implicit surface) output
5) GPU post-processing:
   - marching cubes / iso-surface extraction
   - optional smoothing / decimation
6) Mesh export (OBJ/PLY) and optional renders

### Model strategy options (choose one primary)
- **C1: VRN re-implementation in PyTorch (closest to original)**
  - replicates volumetric regression network
  - highest fidelity to the original method
  - requires careful architectural matching

- **C2: Adopt a modern 3D face reconstruction baseline inspired by VRN**
  - uses a contemporary PyTorch implementation of volumetric/implicit face recon
  - faster to ship, better CUDA compatibility
  - must justify as “modernized implementation” rather than strict VRN replication

> Thesis-friendly approach: implement C2 if schedule is tight; document C1 as long-term extension.

---

## 4) Environment and reproducibility plan

### Option 1 — CUDA Docker (recommended)
Create:
- `docker/Dockerfile.cuda`
- pinned CUDA base image
- pinned PyTorch wheel / version

**Deliverable:**  
```bash
docker build -t vrn-modern:cuda -f docker/Dockerfile.cuda .
docker run --gpus all -v $PWD:/workspace vrn-modern:cuda python run.py ...
```

### Option 2 — Conda environment (fallback)
Create:
- `env/environment.yml`
- a “known-good” setup path

**Deliverable:**  
```bash
conda env create -f env/environment.yml
conda activate vrn-modern
python run.py ...
```

---

## 5) Implementation plan (step-by-step)

### Step C0 — Freeze evaluation subset
Same as Design A/B:
- Choose a stable AFLW2000-3D subset (e.g., 200 images)
- Save filenames list to `docs/aflw2000_subset.txt`

---

### Step C1 — Data pipeline setup (300W-LP + AFLW2000-3D)
Create standardized data layout:
```
datasets/
  300W-LP/
    images/
    annotations/ (if any)
  AFLW2000-3D/
    images/
    gt/ (if available)
```

Write a dataset loader:
- `datasets/aflw2000.py`
- `datasets/300wlp.py`

**Deliverable:** A sanity notebook/script that loads samples and visualizes crops.

---

### Step C2 — Preprocessing modernization
Replace/standardize face detection/alignment:
- Keep it simple and reproducible:
  - face bbox detection
  - landmark alignment (if used)
  - crop + normalize

**Deliverable:** `preprocess.py` that produces deterministic crops.

---

### Step C3 — Model implementation (choose C1 or C2)
#### C1: PyTorch VRN re-implementation
- Implement the volumetric regression network architecture in PyTorch.
- Define output as 3D voxel grid.
- Loss: voxel-wise (e.g., BCE/CE) + optional regularizers.

**Deliverable:** `models/vrn.py` and a forward pass producing volume.

#### C2: Adopt a modern PyTorch baseline
- Select a VRN-like baseline (volumetric or implicit).
- Justify as modernization while keeping the same objective/output.

**Deliverable:** `models/baseline.py` and training/inference scripts.

---

### Step C4 — GPU post-processing & mesh export
- Implement marching cubes on GPU if possible
- Export to OBJ/PLY
- Provide an optional render utility for poster figures

**Deliverable:** `mesh/export.py` and `mesh/marching_cubes.py`

---

### Step C5 — Training / fine-tuning (optional but strong)
- Train or fine-tune on 300W-LP
- Evaluate on AFLW2000-3D subset

**Deliverable:** training logs, checkpoints, and evaluation summary.

---

## 6) Functional verification (“simulation”) for Design C

### Verification goals
- Functional: pipeline runs end-to-end on GPU and outputs valid meshes.
- Comparative: outputs comparable in quality to Design A baseline on same subset.
- Performance: demonstrate measurable speedup and GPU utilization.

### Minimum metrics
- Inference time per image (GPU vs CPU baseline)
- GPU memory usage (VRAM peak)
- Success rate (mesh produced / inputs)
- Qualitative results (mesh renders)

### Optional metrics (if GT is accessible)
- NME (normalized mean error) on landmarks
- Depth/vertex error
- Mesh distance (Chamfer)

---

## 7) Poster assets for Design C

### What to show
- “Modernized CUDA pipeline” diagram (Design C)
- A/B/C comparison panel:
  - input image
  - Design A mesh (CPU legacy)
  - Design B mesh (CUDA post)
  - Design C mesh (CUDA end-to-end)
- Performance table/chart:
  - time/image and GPU utilization

### Minimum poster pack
```
results/poster/designC/
  pipeline_diagram.png
  comparison_grid.png
  timing_table.png
```

---

## 8) Chapter 4 mapping (write-up checklist)

### 4.1 Methodology Overview (Design C)
- Explain modernization approach and why Torch7/CUDA constraints require it.
- Describe full GPU pipeline and reproducible environment.

### 4.2 Preliminary Design / Model Specification (Design C)
- Model definition (architecture choice C1/C2)
- Input/output specs (image → volume/surface → mesh)
- Training strategy (300W-LP) and evaluation (AFLW2000-3D)
- Hardware/software requirements (PyTorch + CUDA, Docker)

---

## 9) Deliverables checklist (Design C)

### Must deliver
- [ ] Reproducible environment (CUDA Docker or conda)
- [ ] Modern preprocessing pipeline (deterministic crops)
- [ ] GPU inference producing 3D outputs + meshes
- [ ] Batch run on AFLW2000 subset
- [ ] Metrics + comparison figures vs Design A/B
- [ ] Chapter 4 Design C section drafted

### Nice-to-have
- [ ] Training/fine-tuning on 300W-LP
- [ ] Quantitative GT-based evaluation
- [ ] Automated poster grid generator

---

## 10) Risk register and mitigation

### Risk C1 — Architecture mismatch vs original VRN
- Mitigation: frame as “modernized implementation inspired by VRN”, validate qualitatively + via metrics.

### Risk C2 — Training time and compute limits
- Mitigation: fine-tune only, or train on a reduced subset; prioritize inference + poster outputs.

### Risk C3 — Marching cubes GPU availability
- Mitigation: keep a CPU fallback; still demonstrate CUDA in model inference and profiling.

---

## 11) Practical timeline (realistic)
- **Week 1:** data loaders + preprocessing + CUDA environment
- **Week 2:** implement model (C1/C2) + inference + mesh export
- **Week 3:** batch experiments + metrics + poster figures + write Chapter 4

---

## 12) Notes for thesis narrative
Design C demonstrates:
- an engineering modernization pathway
- CUDA acceleration aligned with thesis focus
- reproducible experimentation (Docker/conda)
- a clean comparative study vs Design A and Design B

