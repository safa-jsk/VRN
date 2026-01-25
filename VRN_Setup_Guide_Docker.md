# VRN (Design A Baseline) — Setup Guide for Other PCs (Docker)

This guide lets you run the **CPU-only** VRN baseline (Design A) on other computers using the published Docker image **`asjackson/vrn:latest`**.  
It also includes optional notes for GPU/CUDA-ready setups used in later designs.

---

## 0) What you get (expected outputs)

Given an input image like:

- `data/turing.jpg`

The container will produce:

- `data/turing.jpg.crop.jpg` (cropped/aligned face)
- `data/turing.jpg.obj` (3D mesh)

---

## 1) Hardware / OS requirements

### Minimum
- 64-bit CPU (x86_64)
- 8 GB RAM (more is better)
- 5–10 GB free disk space (Docker images + outputs)

### OS
- **Ubuntu 22.04 (recommended)**  
- Windows 11 (works via Docker Desktop, best with WSL2 backend)

> **Note:** The published image is intended for CPU. You do **not** need a GPU for Design A.

---

## 2) Folder layout (portable and recommended)

Create a folder you can copy to other PCs:

```
VRN_Run/
  data/
    in/        # input images
    out/       # outputs (optional organization)
  scripts/     # helper scripts (optional)
  README_run.md
```

---

## 3) Ubuntu 22.04 setup (CPU)

### Step U1 — Install Docker Engine
If Docker isn’t installed yet:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
```

Optional: run Docker without `sudo`:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker --version
docker run --rm hello-world
```

---

## 4) Windows 11 setup (CPU)

### Step W1 — Install Docker Desktop
1. Install **Docker Desktop** on Windows.
2. In Docker Desktop settings:
   - Enable **Use WSL 2 based engine** (recommended)

Verify in PowerShell:
```powershell
docker --version
docker run --rm hello-world
```

### Step W2 — Choose where to run commands
You can run Docker commands in:
- **PowerShell / Windows Terminal**, or
- **WSL2 Ubuntu** (recommended for Linux-like paths/scripts)

> If you use PowerShell, the `-v` mount paths look a bit different. See the examples below.

---

## 5) Running VRN (same idea on all PCs)

### Step R1 — Pull the image
```bash
docker pull asjackson/vrn:latest
```

### Step R2 — Prepare input image folder
From your run folder:
```bash
mkdir -p data
```

Put an image inside `data/` (or `data/in/` if you use that structure).

Example input:
- `data/turing.jpg`

### Step R3 — Run VRN on a single image (Ubuntu / WSL)
From the folder that contains `data/`:

```bash
docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh /data/turing.jpg
```

### Step R4 — Check outputs
```bash
ls -lah data
```

You should see:
- `turing.jpg.crop.jpg`
- `turing.jpg.obj`

---

## 6) Windows path examples (PowerShell)

If your folder is `C:\Users\You\VRN_Run` and `data\turing.jpg` exists:

```powershell
docker run --rm -v ${PWD}\data:/data asjackson/vrn:latest /runner/run.sh /data/turing.jpg
```

If you get mount/path issues, run from WSL2 instead (more reliable for Linux-like scripts).

---

## 7) Batch run template (AFLW2000-3D subset)

### Ubuntu / WSL batch
Put images in:
- `data/in/aflw2000/*.jpg`

Run:
```bash
mkdir -p data/in/aflw2000
mkdir -p data/out/aflw2000

for f in data/in/aflw2000/*.jpg; do
  echo "Processing: $f"
  docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest     /runner/run.sh "/data/in/aflw2000/$(basename "$f")"
done

# collect outputs
mv data/in/aflw2000/*.obj data/out/aflw2000/ 2>/dev/null
mv data/in/aflw2000/*.crop.jpg data/out/aflw2000/ 2>/dev/null
```

---

## 8) Performance logging (optional but useful for thesis)

### Single image timing
```bash
time docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh /data/turing.jpg
```

### Batch timing log
```bash
mkdir -p data/out/aflw2000
for f in data/in/aflw2000/*.jpg; do
  /usr/bin/time -f "%e sec  %M KB  $(basename "$f")"     docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest     /runner/run.sh "/data/in/aflw2000/$(basename "$f")"     2>> data/out/aflw2000/time.log
done
```

---

## 9) Common issues and fixes

### Issue: “Unable to find image 'vrn:latest' locally”
Cause: wrong image name.  
Fix: use the full name:
```bash
asjackson/vrn:latest
```

### Issue: No output mesh created
Check:
- the input file path is correct and accessible in `/data`
- the bind mount is correct:
```bash
docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest ls -lah /data
```

### Issue: Permission problems writing outputs
Fix (Ubuntu):
```bash
sudo chown -R $USER:$USER data
```

### Issue: SELinux `:Z` confusion
Ubuntu usually doesn’t require `:Z`. If you see examples with `:Z`, you can omit it on Ubuntu.

---

## 10) Optional: preparing PCs for CUDA-based designs (B/C)

Design A does **not** require CUDA.  
If later you need GPU containers:

### Ubuntu (GPU Docker prerequisites)
- NVIDIA driver installed and working
- NVIDIA Container Toolkit installed
- Verify:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### Windows 11 (GPU containers)
- Usually easiest via WSL2 + NVIDIA driver support for WSL
- Docker Desktop uses WSL2 backend

> For Design B/C, you’ll likely build a new CUDA image; the current `asjackson/vrn` image is CPU-only.

---

## 11) “Quick Start” checklist for a new PC

1. Install Docker (Ubuntu: `sudo apt install docker.io`; Windows: Docker Desktop)
2. `docker pull asjackson/vrn:latest`
3. Create `data/` and place an image: `data/turing.jpg`
4. Run:
   ```bash
   docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest /runner/run.sh /data/turing.jpg
   ```
5. Confirm: `data/turing.jpg.obj` exists
6. Open in MeshLab for visualization

---

## 12) What to share with teammates/other PCs

- This guide (`VRN_Setup_Guide_Docker.md`)
- A small sample input image set (10–20 images)
- A “known good” command line example and expected output filenames

