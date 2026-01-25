# VRN (Volumetric Regression Network) — Design A Roadmap (Legacy, CPU-only, Docker Baseline)

**Goal:** Establish a *reproducible, CPU-only* baseline pipeline for VRN that generates 3D face meshes (OBJ) from single images, and produces **functional verification artifacts** (outputs + logs + timing) suitable for **Chapter 4 (Part 2)** and your **poster**.

---

## 1) Scope and success criteria

### What Design A includes
- Run VRN inference using the **prebuilt CPU Docker image**: `asjackson/vrn:latest`.
- Process:
  1. Input image → face detection/alignment/crop
  2. Volumetric regression (3D volume)
  3. Isosurface extraction → **mesh** (`.obj`)
- Save and organize outputs and evidence:
  - meshes (`*.obj`)
  - cropped inputs (`*.crop.jpg`)
  - runtime logs + timing stats

### What Design A does *not* include
- No CUDA/GPU acceleration.
- No code modernization or framework port.
- No retraining on 300W-LP (baseline is inference-only).

### Success criteria (Design A is “done” when…)
- ✅ Can generate `.obj` for a set of test images **consistently**.
- ✅ Outputs are organized and repeatable via scripts/commands.
- ✅ Baseline metrics captured (runtime / throughput / success rate).
- ✅ Poster-ready mesh renders produced (screenshots/renders).

---

## 2) Environment baseline (record in thesis)

### Operating system
- **Ubuntu 22.04** recommended for the baseline run (stable Docker behavior).

### Runtime dependencies
- Docker Engine + permission to run containers.
- The VRN CPU image:
  - `asjackson/vrn:latest` (contains `/runner/run.sh` inside the image)

### Minimal folder structure (host)
```
VRN/
  data/
    in/        # input images
    out/       # outputs (meshes, crops, logs)
    tmp/       # optional scratch
  scripts/     # your helper scripts for repeatability
  results/     # curated meshes + poster figures + metrics
  docs/        # report notes and experiment logs
```

---

## 3) Baseline “known good” single-image run

### Step A1 — Prepare an input image
Put a test image in `data/in/`:
- Example: `data/in/turing.jpg`

### Step A2 — Run VRN inside Docker
From the repo root:
```bash
docker pull asjackson/vrn:latest
docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh /data/in/turing.jpg
```

### Step A3 — Verify outputs
Expected outputs (same directory as input):
- `turing.jpg.crop.jpg`
- `turing.jpg.obj`

Check:
```bash
ls -lah data/in
```

**Evidence to save (screenshots/logs):**
- terminal output showing “Regressed 3D volume” and “Calculated the isosurface”
- MeshLab screenshot showing the resulting mesh

---

## 4) Batch processing pipeline (AFLW2000-3D evaluation subset)

### Why AFLW2000-3D here?
- Use AFLW2000-3D as **test/evaluation** set for Design A outputs.
- For the poster: pick representative samples with different poses.

### Step A4 — Populate evaluation inputs
Create input folder:
```bash
mkdir -p data/in/aflw2000
```

Copy a subset of AFLW2000-3D images into `data/in/aflw2000/`.

### Step A5 — Run batch inference
Create output folder:
```bash
mkdir -p data/out/aflw2000
```

Run:
```bash
for f in data/in/aflw2000/*.jpg; do
  echo "Processing: $f"
  docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh "/data/in/aflw2000/$(basename "$f")"
done
```

### Step A6 — Collect outputs cleanly
Move meshes and crops to `data/out/aflw2000/`:
```bash
mv data/in/aflw2000/*.obj data/out/aflw2000/ 2>/dev/null
mv data/in/aflw2000/*.crop.jpg data/out/aflw2000/ 2>/dev/null
```

---

## 5) Functional verification (“simulation”) for Design A

Design A verification focuses on **functional correctness** and **baseline performance**.

### Verification checklist
- **Output existence:** each input should produce `*.obj` (and usually `*.crop.jpg`)
- **Mesh sanity checks (visual):**
  - mesh is not empty / not exploded
  - face geometry is plausible (nose/eyes region visible)
- **Failure modes captured:**
  - extreme side poses not detected (dlib limitations)
  - unusual lighting/occlusion failures

### Quantitative metrics (baseline)
#### Runtime per image
Use `time`:
```bash
time docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh /data/in/turing.jpg
```

For batch, record per-image time:
```bash
for f in data/in/aflw2000/*.jpg; do
  /usr/bin/time -f "%e sec  %M KB  $(basename "$f")"     docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest     /runner/run.sh "/data/in/aflw2000/$(basename "$f")"     2>> data/out/aflw2000/time.log
done
```

#### Success rate
Count inputs vs outputs:
```bash
num_in=$(ls -1 data/in/aflw2000/*.jpg 2>/dev/null | wc -l)
num_out=$(ls -1 data/out/aflw2000/*.obj 2>/dev/null | wc -l)
echo "Success: $num_out / $num_in"
```

Save this summary to `docs/designA_metrics.md`.

---

## 6) Poster-ready assets (must-have)

### Step A7 — Select representative meshes
Pick 6–10 examples:
- frontal
- mild yaw
- large yaw
- different lighting / background
- 1–2 failure cases (optional but informative)

### Step A8 — Render/screenshot in MeshLab
For each mesh:
- consistent camera angle (front / 3-4 view / side)
- consistent lighting style
- export screenshots at high resolution

Save to:
```
results/poster/meshes/
  mesh_01_front.png
  mesh_01_34.png
  mesh_01_side.png
```

### Step A9 — Create a small “results grid”
Make a 2×3 or 3×3 grid image for the poster:
- input image on top, reconstructed mesh below

(We can automate this later, but manual is fine for Design A.)

---

## 7) What to write in Chapter 4 for Design A

### 4.1 Methodology Overview (Design A section)
Include:
- baseline pipeline diagram (Input → Crop/Align → Volume Regression → Isosurface → Mesh)
- environment and constraints (CPU-only Docker)
- why baseline is needed (reference point for later designs)

### 4.2 Preliminary Design / Model Specification (Design A)
Specify:
- input format (JPEG face image)
- output format (`*.obj` mesh, `*.crop.jpg`)
- containerized execution (`asjackson/vrn:latest`, `/runner/run.sh`)
- verification plan and metrics captured

---

## 8) Deliverables checklist (Design A)

### Must deliver
- [ ] `data/` folder structure established
- [ ] single-image demo verified (`turing.jpg.obj`)
- [ ] batch run completed on AFLW2000-3D subset
- [ ] `time.log` + success-rate summary saved
- [ ] 6–10 poster-ready mesh screenshots exported
- [ ] Chapter 4 sections drafted for Design A (Methodology + Spec)

### Nice-to-have
- [ ] brief failure analysis with example images
- [ ] automated screenshot/render pipeline

---

## 9) Recommended timeline (practical)
- **Day 1:** Single image + folder structure + 10-image batch
- **Day 2:** 50–200 AFLW2000 images batch + timing + success-rate
- **Day 3:** Poster screenshots + results grid + Chapter 4 write-up

---

## 10) Notes for continuity into Design B/C
Design A output artifacts become the baseline comparison for later:
- same test subset (AFLW2000-3D)
- same success-rate criteria
- same runtime measurement method
- same poster mesh selection set (compare output quality visually)
