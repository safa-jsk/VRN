# VRN — Design B Roadmap (CUDA-Accelerated Variant)

**Goal (Design B):** Introduce **measurable CUDA/GPU acceleration** into the VRN pipeline while keeping the **core legacy VRN model unchanged** (same weights, same outputs *as much as possible*).  
This design is deliberately engineered to be **feasible on modern GPUs (e.g., RTX 2050)**, even though the original VRN Torch7 stack expects very old CUDA/cuDNN versions.

> **Key idea:** Keep VRN inference as-is (CPU baseline / legacy execution), but accelerate the **post-processing** (volume→mesh and related steps) on GPU using a modern CUDA-capable stack.

---

## 1) Why Design B (rationale for Chapter 4)

### Motivation
- VRN is legacy (Torch7-era). Full Torch7 GPU execution generally requires **CUDA 7.5/8.0 + cuDNN 5.1**, which is not practical on modern RTX GPUs.
- However, VRN’s pipeline has a clear **post-processing bottleneck**:
  - isosurface extraction (Marching Cubes) and optional smoothing / rendering

### Design B objective
- Achieve **speedup** and **GPU utilization** by moving:
  - **Isosurface extraction** (Marching Cubes) to CUDA
  - optional mesh post-processing (smoothing, decimation) to GPU-capable tools where possible
- Preserve Design A’s reproducibility by keeping:
  - identical inputs
  - same dataset subset
  - same output formats (OBJ/PLY)

---

## 2) High-level architecture

### Pipeline comparison
**Design A (baseline):**  
Input → crop/align → VRN volume regression → (CPU) isosurface → mesh

**Design B (CUDA variant):**  
Input → crop/align → VRN volume regression → **export volume** → (GPU/CUDA) marching cubes → mesh (+ optional post)

### Execution model
- VRN inference remains containerized (Design A container or a derived one)
- Post-processing runs in a modern CUDA environment (PyTorch / CUDA toolkit)

---

## 3) Design B variants (choose one primary; keep the other as fallback)

### B1 (Recommended): CPU VRN + GPU Marching Cubes (modern PyTorch CUDA)
- Extract the **3D volume** from VRN output.
- Run **GPU marching cubes** using a modern library stack:
  - PyTorch + CUDA
  - `pytorch3d` (if available) or CUDA-accelerated marching cubes alternatives
- Write mesh to `.obj` or `.ply`.

**Pros:** feasible on RTX 2050, clear speedup, modern tooling, strong CUDA story  
**Cons:** requires modifying runner to save volume in a readable format

### B2 (Risky): Full Torch7 GPU enablement
- Build a CUDA-enabled Torch7 container and run VRN inference on GPU.

**Pros:** “pure” GPU VRN inference  
**Cons:** very likely incompatible on modern GPUs due to ancient CUDA/cuDNN expectations; high schedule risk

> **Recommendation for thesis:** Implement **B1** and document B2 as “attempted / infeasible due to version constraints” (still valuable engineering analysis).

---

## 4) Implementation plan (B1 recommended)

### Step B0 — Establish baseline parity dataset
- Freeze the exact test subset used in Design A:
  - e.g., 50–200 images from AFLW2000-3D
- Save list of filenames as `docs/aflw2000_subset.txt`.

---

### Step B1 — Modify the VRN runner to export the predicted volume
You need a **volume export** step. Options depend on what the container already produces internally:

1) If VRN writes a volume file already:
- Locate it in container output and mount it out.

2) If not:
- Add a step in the runner pipeline to save:
  - raw voxel grid as `.npy` (preferred), or
  - `.mat` / `.h5` / `.bin`

**Deliverable:** For each input image, you get:
- `image.crop.jpg`
- `image.vol.npy` (or equivalent)
- (optional) legacy `image.obj` from CPU for comparison

---

### Step B2 — Build a CUDA post-processing environment
Create a new folder (host repo):
```
cuda_post/
  env/requirements.txt
  marching_cubes_cuda.py
  io.py
  benchmarks.py
```

Use a modern Python + CUDA stack (Ubuntu):
- Python 3.10+ recommended
- PyTorch with CUDA (matching your driver)
- A marching cubes implementation with GPU support

**Deliverable:** A script:
```bash
python cuda_post/marching_cubes_cuda.py --in data/out/aflw2000/*.vol.npy --out results/designB_meshes/
```

---

### Step B3 — Integrate into a reproducible “two-stage” pipeline
Create a script like:
- `scripts/designB_run.sh`

Pseudo-flow:
1) Run VRN container to generate crop + volume
2) Run CUDA script to produce meshes
3) Store outputs in `results/designB/`

**Deliverable:** One command to reproduce results:
```bash
./scripts/designB_run.sh data/in/aflw2000 results/designB
```

---

## 5) Functional verification (“simulation”) for Design B

### Verification goals
- **Correctness:** GPU-generated mesh should be consistent with baseline (Design A) mesh quality.
- **Performance:** show CUDA acceleration and measurable speedup in post-processing.

### Verification checklist
- Output existence: each input has:
  - `*.vol.npy`
  - `*.obj` (GPU-generated)
- Mesh sanity checks:
  - no empty mesh
  - geometry is plausible
- Consistency checks (optional but strong):
  - Compare number of vertices/faces vs Design A
  - Compare bounding box size
  - Compute Chamfer distance if both meshes exist (advanced)

### Quantitative metrics (minimum)
- Time per image for:
  - Volume export stage (CPU)
  - GPU marching cubes stage
- GPU utilization and memory:
  - `nvidia-smi` snapshots during batch run

**Record in:** `results/designB/metrics/`

---

## 6) Poster assets for Design B

### What to show
- A/B visual comparison:
  - input image
  - Design A mesh
  - Design B mesh
- Performance comparison chart:
  - CPU isosurface time vs GPU isosurface time
  - (optionally) total pipeline time if meaningful

### Minimum poster pack
```
results/poster/designB/
  mesh_comparisons/
  timing_plot.png
  pipeline_diagram.png
```

---

## 7) Chapter 4 mapping (write-up checklist)

### 4.1 Methodology Overview (Design B section)
Include:
- Two-stage pipeline diagram
- Motivation: modern GPU constraints vs legacy Torch7 requirements
- Design decision: accelerate post-processing (marching cubes) with CUDA

### 4.2 Preliminary Design / Model Specification (Design B)
Specify:
- Inputs: same as Design A
- Intermediate: voxel volume file format (`.npy` recommended)
- Outputs: `.obj/.ply`
- CUDA environment:
  - GPU requirements (RTX 2050)
  - PyTorch CUDA version used
- Verification plan and metrics

---

## 8) Deliverables checklist (Design B)

### Must deliver
- [ ] VRN runner exports volume for each image (e.g., `*.vol.npy`)
- [ ] CUDA marching cubes script produces meshes (`*.obj/.ply`)
- [ ] Batch run works on AFLW2000 subset
- [ ] Timing + GPU utilization logs captured
- [ ] A vs B visual comparison figures for poster
- [ ] Chapter 4 Design B section drafted (method + spec + verification)

### Nice-to-have
- [ ] Mesh similarity metric (Chamfer / ICP) between A and B outputs
- [ ] Automated poster grid generation

---

## 9) Risk register (include in thesis constraints)

### Risk R1 — Full Torch7 GPU infeasible on modern RTX GPUs
- Mitigation: choose B1 (post-processing CUDA), document B2 as infeasible due to dependency constraints.

### Risk R2 — Difficulty exporting volume from legacy runner
- Mitigation: modify container runner; if too hard, replicate CPU marching cubes outside and then swap it for GPU marching cubes.

### Risk R3 — GPU marching cubes library availability
- Mitigation: keep a CPU marching cubes script in Python as fallback to validate volume format first.

---

## 10) Practical timeline
- **Day 1:** Identify where volume exists / modify runner to export volume
- **Day 2:** Implement marching cubes pipeline (start CPU, then CUDA)
- **Day 3:** Batch run + metrics + poster figures + Chapter 4 text

---

## 11) Notes for continuity into Design C
Design C can evolve from B by:
- moving VRN inference to a modern framework (PyTorch) for full GPU acceleration, or
- re-training/fine-tuning on 300W-LP, keeping the B pipeline for evaluation.

