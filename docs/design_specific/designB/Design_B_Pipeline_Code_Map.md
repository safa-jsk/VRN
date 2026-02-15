# Design B: Implementation-Accurate Pipeline Code Map

**Generated:** 2026-01-31  
**Focus:** Design B GPU-accelerated implementation with exact file/function/line references

---

## A) ENTRY POINTS + EXECUTION

### Step 1: Actual Inference Entry Point (Design B Stage 2: GPU Marching Cubes)

**What happens:** Load pre-exported VRN volumes (.npy files), run CUDA marching cubes, apply post-processing, export meshes

**Why:** Stage 1 (VRN inference) runs in legacy Docker (CPU-only). Stage 2 demonstrates pure CUDA acceleration on pre-extracted volumes.

**How:**

- Loads 200×192×192 boolean volumes from `.npy` files
- Transfers to GPU as float32 tensor
- Calls custom CUDA marching cubes kernel
- Post-processes: axis transform, Z-scaling, vertex merge (tolerance=0.1), RGB color mapping
- Exports as Wavefront .obj

**Where:**

- **Main script:** `designB/python/marching_cubes_cuda.py :: main block :: lines 335-385`
- **Single volume processing:** `designB/python/marching_cubes_cuda.py :: process_volume_to_mesh() :: lines 111-219`
- **Batch processing:** `designB/python/marching_cubes_cuda.py :: batch_process_volumes() :: lines 223-320`

**Execution command:**

```bash
# Single volume
python3 designB/python/marching_cubes_cuda.py \
  --input data/out/designB/volumes/image00002.npy \
  --output data/out/designB/meshes/image00002.obj \
  --threshold 0.5

# Batch processing
python3 designB/python/marching_cubes_cuda.py \
  --input data/out/designB/volumes \
  --output data/out/designB/meshes \
  --threshold 0.5 \
  --pattern '*.npy'
```

---

### Step 2: Actual Evaluation/Benchmark Entry Point

**What happens:** Benchmark GPU vs CPU marching cubes on 43 AFLW2000 volumes, measure timing (3 runs per volume), generate plots

**Why:** Validate 18.36× speedup claim, measure consistency (std dev), produce performance data for thesis

**How:**

- Loads each .npy volume
- Runs CPU marching cubes (scikit-image) 3 times, records min/avg/std
- Runs GPU marching cubes (custom CUDA) 3 times with `torch.cuda.synchronize()` for accurate timing
- Computes speedup = CPU_time / GPU_time
- Saves benchmark_results.json with per-volume stats
- Generates timing_comparison.png and speedup_chart.png

**Where:**

- **Main benchmark script:** `designB/python/benchmarks.py :: main block :: lines 287-318`
- **Single volume benchmark:** `designB/python/benchmarks.py :: benchmark_single_volume() :: lines 25-105`
- **Batch benchmark:** `designB/python/benchmarks.py :: benchmark_batch() :: lines 148-213`
- **Plot generation:** `designB/python/benchmarks.py :: plot_benchmark_results() :: lines 216-284`

**Execution command:**

```bash
# Wrapper script
./designB/scripts/run_benchmarks.sh

# Or direct Python
python3 designB/python/benchmarks.py \
  --volumes data/out/designB/volumes \
  --output data/out/designB/benchmarks_cuda \
  --runs 3 \
  --plot
```

---

### Step 3: Config/Args Parsing and Defaults

**What happens:** Command-line argument parsing for inference and benchmarking scripts

**Why:** Allow flexible threshold, device selection, input/output paths

**How (marching_cubes_cuda.py):**

- `--input` (required): volume file or directory
- `--output` (required): mesh file or directory
- `--threshold` (default: **0.5**): isosurface threshold for boolean volumes
- `--cpu`: force CPU mode (disable GPU)
- `--pattern` (default: `*.npy`): glob pattern for batch mode
- `--image-dir`: optional directory for RGB color mapping

**How (benchmarks.py):**

- `--volumes` (required): directory containing .npy volumes
- `--output` (default: `data/out/designB/benchmarks`): output directory
- `--runs` (default: **3**): number of timing runs per volume
- `--max-volumes`: limit number of volumes
- `--plot`: generate plots from results

**Where:**

- **Inference args:** `designB/python/marching_cubes_cuda.py :: argparse setup :: lines 335-349`
- **Benchmark args:** `designB/python/benchmarks.py :: argparse setup :: lines 287-304`

**Key defaults:**

- Threshold: **0.5** (matches VRN's boolean volume extraction)
- Runs: **3** (for statistical stability)
- Merge tolerance: **0.1** (hardcoded in `volume_io.py :: save_mesh_obj() :: line 110`, design choice for higher quality vs Design A's 1.0)

---

## B) GPU EXECUTION AND PERFORMANCE FLAGS (Design B Speed Layer)

### Step 4: CUDA Device Selection and Enforcement

**What happens:** Checks CUDA availability, selects GPU device, transfers tensors to GPU

**Why:** Ensure CUDA is available before attempting GPU execution, fail gracefully to CPU if not

**How:**

- `torch.cuda.is_available()` check at multiple points
- Device selection: hardcoded `device='cuda'` (uses default GPU 0)
- Tensor transfer: `volume_to_tensor(volume, device='cuda')` converts NumPy → torch.Tensor on GPU

**Where:**

- **Device check:** `designB/python/marching_cubes_cuda.py :: process_volume_to_mesh() :: lines 152-156`
- **Tensor transfer:** `designB/python/volume_io.py :: volume_to_tensor() :: lines 76-89`
- **GPU availability log:** `designB/python/benchmarks.py :: benchmark_single_volume() :: line 71`
- **CUDA ext check:** `designB/cuda_kernels/cuda_marching_cubes.py :: module init :: lines 12-17`

**Enforcement:**

- If `torch.cuda.is_available() == False`, prints warning and falls back to CPU
- No multi-GPU support (always uses device 0)
- CUDA extension import failure triggers fallback to scikit-image CPU

**Missing:** No explicit device_id selection for multi-GPU systems (would need `--device-id` arg and `torch.cuda.device(id)` context)

---

### Step 5: cuDNN / TF32 / AMP Flags

**What happens:** **NONE** - No explicit cuDNN, TF32, or AMP configuration

**Why:** Custom CUDA kernel bypasses PyTorch's autograd/cuDNN; marching cubes is a pure geometry operation (no neural network)

**Where:** Missing

**Note:** VRN inference (Stage 1) uses Torch7 with cuDNN 5.1, but that's in the Docker container (Design A). Design B Stage 2 has zero PyTorch autograd operations, so cuDNN/AMP are irrelevant.

**Closest related code:**

- CUDA kernel compilation flags: `designB/setup.py :: extra_compile_args :: lines 34-43`
  - `--use_fast_math`: enables fast FP32 approximations (not TF32)
  - `-O3`: maximum compiler optimization
  - No explicit tensor core or FP16 usage

**What I should add (if needed for future):**

- FP16/TF32 support in kernel (would require changing kernel to accept half-precision input)
- Currently irrelevant since marching cubes operates on boolean/float32 volumes (no training)

---

### Step 6: GPU Warmup Iterations

**What happens:** **NO EXPLICIT WARMUP** in production code

**Why:** Benchmarking best practices require warmup to initialize CUDA context, load kernels, stabilize GPU clocks

**Where:** **Missing** from `designB/python/benchmarks.py`

**Closest related code:**

- Benchmark runs 3 iterations per volume: `designB/python/benchmarks.py :: benchmark_single_volume() :: lines 75-88`
- No separate warmup phase before timing

**What I should add:**

```python
# In benchmark_single_volume(), before timing loop:
# Warmup: 2-3 dummy runs to initialize CUDA
for _ in range(3):
    _ = marching_cubes_gpu_pytorch(volume_gpu, threshold)
torch.cuda.synchronize()
# Then start actual timing
```

**Note:** Documentation claims warmup (Chapter 4.1 line 299, 373), but code does NOT implement it. This is a **documentation-code inconsistency**.

---

### Step 7: torch.inference_mode() / no_grad()

**What happens:** **NONE** - No gradient disabling

**Why:** Marching cubes is not part of a neural network, no gradients computed. Custom CUDA kernel bypasses autograd entirely.

**Where:** Missing (intentionally unnecessary)

**Closest related code:**

- CUDA kernel is pure C++/CUDA, no PyTorch autograd: `designB/cuda_kernels/marching_cubes_kernel.cu`
- Volume tensor is `.float()` but never requires_grad: `designB/python/volume_io.py :: volume_to_tensor() :: line 84`

**Why not needed:**

- `torch.from_numpy(volume).float()` creates tensor with `requires_grad=False` by default
- No backward pass ever called
- Custom CUDA op via pybind11 doesn't register in autograd graph

**What I should add (best practice, minimal overhead):**

```python
# In marching_cubes_gpu_pytorch():
@torch.inference_mode()  # or with torch.no_grad():
def marching_cubes_gpu_pytorch(volume_tensor, threshold=0.5):
    ...
```

---

### Step 8: GPU Synchronization for Timing

**What happens:** `torch.cuda.synchronize()` called before and after GPU operations to ensure accurate timing

**Why:** CUDA kernels are asynchronous. Without synchronization, `time.time()` would return before kernel completion, giving falsely fast timings.

**How:**

1. Transfer volume to GPU
2. **`torch.cuda.synchronize()`** ← wait for transfer
3. Start timer: `t_start = time.time()`
4. Launch CUDA marching cubes
5. **`torch.cuda.synchronize()`** ← wait for kernel completion
6. Stop timer: `t_end = time.time()`
7. Elapsed = `t_end - t_start`

**Where:**

- **Benchmark timing:** `designB/python/benchmarks.py :: benchmark_single_volume() :: lines 76, 81, 86`
  ```python
  torch.cuda.synchronize()  # Line 76: after GPU transfer
  torch.cuda.synchronize()  # Line 81: before timing
  torch.cuda.synchronize()  # Line 86: after kernel
  ```
- **CUDA kernel wrapper:** `designB/cuda_kernels/marching_cubes_bindings.cpp :: marching_cubes_forward() :: line 56`
  ```cpp
  cudaStreamSynchronize(stream);  // C++ sync after kernel launch
  ```

**Timing method:**

- Python standard library `time.time()` (wall-clock time)
- **No CUDA events:** Could use `torch.cuda.Event()` for more precise GPU-only timing (excludes Python overhead)

**What I should add (for even more precise GPU-only timing):**

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
marching_cubes_gpu_pytorch(volume_gpu, threshold)
end_event.record()
torch.cuda.synchronize()
gpu_time = start_event.elapsed_time(end_event) / 1000.0  # ms to seconds
```

---

## C) VRN OUTPUT → MESH POST-PROCESSING (Poster-Critical)

### Step 9: VRN Volume/Occupancy Tensor Production

**What happens:** VRN produces 200×192×192 voxel volume representing 3D occupancy

**Why:** VRN's CNN regresses volumetric face representation; marching cubes extracts surface from this volume

**How (Design A, in Docker):**

- VRN Torch7 model outputs tensor of shape (200, 192, 192)
- Original VRN: `vol = vol:byte()` (converts to uint8, 0-255 range)
- Script threshold: `mcubes.marching_cubes(vol, 10)` in `raw2obj.py :: line 23`

**How (Design B, volume export):**

- Volume exported from Docker as `.raw` binary (200×192×192 int8)
- Converted to `.npy` for Python: `designB/python/convert_raw_to_npy.py`
- Loaded as boolean: `vol.astype(bool)` in `volume_io.py :: load_volume_npy() :: line 67`

**Where:**

- **VRN output (Design A):** Inside Docker container, not directly accessible
  - Reference: `raw2obj.py :: lines 18-23` (original VRN script)
    ```python
    vol = np.fromfile(args.volume, dtype=np.int8)
    vol = vol.reshape((200,192,192))
    vol = vol.astype(float)
    ```
- **Design B volume loader:** `designB/python/volume_io.py :: load_volume_npy() :: lines 56-67`
- **Design B volume format:** `.npy` files, shape (200, 192, 192), dtype after loading: **bool**

**Tensor properties:**

- **Name:** `volume` (Python variable)
- **Shape:** `(200, 192, 192)` ← Z, Y, X indexing (NumPy convention)
- **Dtype (raw VRN):** int8 (values 0 or non-zero)
- **Dtype (Design B):** bool after `vol.astype(bool)` conversion
- **Device:** CPU (NumPy array) → converted to `torch.Tensor` on GPU via `volume_to_tensor()`

---

### Step 10: Volume Thresholding and Smoothing

**What happens:** Boolean thresholding applied to volume before marching cubes

**Why:** VRN outputs continuous occupancy; thresholding creates binary inside/outside classification for isosurface extraction

**How:**

- **Design A (original VRN):** Uses threshold=**10** on float volume in `raw2obj.py :: line 23`
  ```python
  vertices, triangles = mcubes.marching_cubes(vol, 10)  # Threshold 10 on int8-casted-to-float
  ```
- **Design B:** Uses threshold=**0.5** on boolean volume
  - Volume converted to bool: `vol.astype(bool)` (0 → False, non-zero → True)
  - Marching cubes threshold: **0.5** (middle value for boolean treated as float)

**Where:**

- **Volume bool conversion:** `designB/python/volume_io.py :: load_volume_npy() :: line 67`
  ```python
  if vol.dtype != bool:
      vol = vol.astype(np.int8).astype(bool)
  ```
- **Threshold parameter (Design B):**
  - **Default:** `marching_cubes_cuda.py :: argparse default :: line 342` → `default=0.5`
  - **Function signature:** `marching_cubes_gpu_pytorch(volume_tensor, threshold=0.5) :: line 37`
  - **CUDA kernel:** `marching_cubes_kernel.cu :: isolevel parameter :: line 46`

**Smoothing:** **NONE** - No Gaussian blur or morphological operations applied

**Key fix from bug investigation:**

- **Original bug:** Design B initially used threshold=10.0 on boolean volumes (wrong!)
- **Fix:** Changed to threshold=0.5 for boolean volumes (matches VRN's binary semantics)
- **Documentation:** Chapter 4.1 line 283 mentions "Threshold Value Bug"

---

### Step 11: CUDA Marching Cubes Call Site

**What happens:** Custom CUDA kernel extracts isosurface from 3D volume

**Why:** 18.36× speedup over CPU (scikit-image), demonstrates CUDA expertise for thesis

**How:**

- Allocate output buffers on GPU (vertices, triangles, counters)
- Launch CUDA kernel with 8×8×8 thread blocks
- Each thread processes one voxel cube, generates 0-5 triangles
- Atomic counters track vertex/triangle counts
- Return trimmed tensors to CPU

**Where:**

**File/Module:** `designB/cuda_kernels/`

**CUDA kernel implementation:**

- **Kernel:** `marching_cubes_kernel.cu :: marchingCubesKernel() :: lines 40-141`
- **Kernel launcher:** `marching_cubes_kernel.cu :: launchMarchingCubes() :: lines 150-181`

**Python wrapper:**

- **PyBind11 binding:** `marching_cubes_bindings.cpp :: marching_cubes_forward() :: lines 23-58`
- **Python interface:** `cuda_marching_cubes.py :: marching_cubes_gpu() :: lines 21-72`

**Call site in pipeline:**

- **High-level wrapper:** `marching_cubes_cuda.py :: marching_cubes_gpu_pytorch() :: lines 37-76`
  - Line 54: `verts, faces = marching_cubes_gpu(volume_tensor, isolevel=threshold, device='cuda')`
- **Process volume:** `marching_cubes_cuda.py :: process_volume_to_mesh() :: line 168`
  - Uses `marching_cubes_gpu_pytorch()` if GPU enabled

**Input/Output tensor shapes:**

- **Input:** `volume_tensor` of shape `(200, 192, 192)`, dtype `float32`, device `cuda`
- **Output (raw):**
  - `vertices`: `(N, 3)` float32, N ≈ 145,000–220,000 (pre-merge)
  - `faces`: `(M, 3)` int32, M ≈ 48,000–73,000 (pre-merge)
- **Output (post-processed):**
  - Vertices: ~63,571 (Design B) vs ~37,587 (Design A) after merge

**Grid/Block config:**

- **Thread block:** `dim3(8, 8, 8)` → 512 threads per block
- **Grid:** `((dimX+7)/8, (dimY+7)/8, (dimZ+7)/8)` → e.g., (25, 24, 24) for 200×192×192
- **Where:** `marching_cubes_kernel.cu :: launchMarchingCubes() :: lines 155-161`

---

### Step 12: Coordinate Fixes (Axis Swap, Z-Scale, Normalization)

**What happens:** Transform raw marching cubes coordinates to match VRN's expected coordinate system

**Why:** VRN volume uses Z,Y,X indexing; final mesh needs X,Y,Z orientation. Z-axis scaling corrects aspect ratio.

**How:**

1. **Axis permutation:** Swap (Z, Y, X) → (X, Y, Z)
   - `vertices[:, [2, 1, 0]]` ← swap first and last columns
2. **Z-scaling:** Compress depth by factor of 0.5
   - `vertices[:, 2] *= 0.5`
3. **No normalization:** No centering or unit-scale normalization applied

**Where:**

- **Post-processing:** `designB/python/volume_io.py :: save_mesh_obj() :: lines 99-105`
  ```python
  if apply_vrn_transform:
      vertices_transformed = vertices.copy()
      # Swap axes: (x,y,z) -> (z,y,x)
      vertices_transformed = vertices_transformed[:, [2, 1, 0]]  # Line 102
      # Scale Z-axis
      vertices_transformed[:, 2] *= 0.5  # Line 104
  ```

**Reference (Design A):**

- Original VRN: `raw2obj.py :: lines 23-24`
  ```python
  vertices = vertices[:,(2,1,0)]  # Axis swap
  vertices[:,2] *= 0.5            # Z-scale
  ```

**Critical ordering:** Transform happens **before** vertex merging and RGB color mapping (lines 107-111)

---

### Step 13: Mesh Cleanup (Vertex Merge, Degenerate Face Removal)

**What happens:** Merge duplicate vertices to reduce mesh size and improve topology

**Why:** Marching cubes generates ~172K vertices with many duplicates; merging produces cleaner, smaller mesh

**How:**

- Use `trimesh.Trimesh.merge_vertices()` with tolerance parameter
- **Design A:** tolerance = **1.0** (aggressive, reduces to ~37.6K vertices)
- **Design B:** tolerance = **0.1** (preserves detail, reduces to ~63.6K vertices)
- Trimesh automatically removes degenerate faces during merge

**Where:**

- **Merge operation:** `designB/python/volume_io.py :: save_mesh_obj() :: lines 107-111`
  ```python
  mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces)
  # Use smaller merge tolerance for smoother mesh (VRN uses 1, but we use 0.1 for better quality)
  trimesh.constants.tol.merge = 0.1  # Line 110 ← KEY DIFFERENCE FROM DESIGN A
  mesh.merge_vertices()              # Line 111
  ```

**Degenerate face removal:** **Implicit** in `trimesh.merge_vertices()` (removes zero-area triangles)

**Design A vs Design B:**

- **Design A (original VRN):** No explicit tolerance visible in `raw2obj.py`, likely uses trimesh default (~1e-8) or VRN's internal merge with tolerance=1.0
- **Design B:** Explicit `trimesh.constants.tol.merge = 0.1` setting

**Effect on quality:**

- Design A: 37,587 vertices (more aggressive merge)
- Design B: 63,571 vertices (+69% higher resolution, smoother surfaces)

---

### Step 14: Color Mapping (Vertex Colors / Texture)

**What happens:** Map RGB colors from input image to mesh vertices

**Why:** Produce photorealistic colored 3D face meshes for visualization

**How:**

1. Load input image, resize to 192×192
2. For each vertex, map X,Y coordinates (after transformation) to image pixels
3. Use nearest-neighbor lookup: `img_array[y_img, x_img, :3]`
4. Assign RGB to vertex

**Where:**

- **Color mapping:** `designB/python/volume_io.py :: save_mesh_obj() :: lines 113-131`
  ```python
  if image_path is not None:
      img = Image.open(image_path)
      img = img.resize((192, 192))  # Line 116
      img_array = np.array(img)

      # Map transformed vertex X,Y coordinates to image RGB (nearest neighbor)
      x_img = np.clip(mesh.vertices[:, 0].astype(int), 0, 191)  # Line 121
      y_img = np.clip(mesh.vertices[:, 1].astype(int), 0, 191)  # Line 122

      # Extract RGB from image
      if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
          vertex_colors = img_array[y_img, x_img, :3]  # Line 126
          mesh.visual.vertex_colors = vertex_colors     # Line 127
  ```

**Timing:** Colors mapped **AFTER** transformation and vertex merging (critical for correct correspondence)

**Color source:** Input 2D image provided via `--image-dir` parameter

**Design A reference:** `raw2obj.py :: lines 25-38` (similar nearest-neighbor approach)

**Texture mapping:** **NONE** - Only per-vertex colors, no UV mapping

---

### Step 15: Mesh Export (OBJ/PLY Writing, Output Naming, Render/Turntable)

**What happens:** Write mesh to Wavefront .obj file with vertex colors

**Why:** Standard format for MeshLab, Blender, thesis figures

**How:**

- Use `trimesh.export()` function
- .obj format with vertex colors: `v x y z r g b`
- Face connectivity: `f i j k` (1-indexed)

**Where:**

- **Export function:** `designB/python/volume_io.py :: save_mesh_obj() :: lines 133-137`

  ```python
  mesh.export(output_path)  # Line 133

  print(f"Saved mesh: {output_path}")  # Line 135
  print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")  # Line 136
  ```

**Output directory naming:**

- Single file: User-specified path via `--output` arg
- Batch mode: `output_dir / (volume_stem + '.obj')`
  - Example: `image00002.npy` → `image00002.obj`
  - **Where:** `marching_cubes_cuda.py :: batch_process_volumes() :: line 269`

**Render/Turntable generation:** **NOT IMPLEMENTED** in automatic pipeline

**Closest related:**

- Manual visualization script exists: `vis.py` (renders volume, not final mesh)
- Poster figure generation: `designB/python/generate_poster_figures.py` (exists but not documented in this map)

**What I should add:**

- Automated screenshot generation (MeshLab command-line: `meshlabserver -i mesh.obj -o screenshot.png -s render_script.mlx`)
- Turntable video generation (Open3D or PyVista for 360° rotation)

---

## D) VERIFICATION HOOKS

### Step 16: Speed Metrics Computation and Logging

**What happens:** Measure and log marching cubes timing, compute speedup, save to JSON/CSV

**Why:** Validate performance claims, generate data for thesis tables/plots

**How:**

- Per-volume timing: `time.time()` before/after marching cubes
- Compute speedup: `cpu_time / gpu_time`
- Aggregate stats: mean, std, min, max across all volumes
- Save JSON with per-volume results and summary

**Where:**

**Speed metric computation:**

- **Benchmark script:** `designB/python/benchmarks.py`
  - Per-volume timing: `benchmark_single_volume() :: lines 62-87`
  - Speedup calc: `results['speedup'] = results['cpu_time_min'] / results['gpu_time_min']` :: line 95
  - Summary stats: `benchmark_batch() :: lines 166-176`

**Logging/Output:**

- **JSON export:** `benchmarks.py :: benchmark_batch() :: lines 195-208`
  - File: `data/out/designB/benchmarks_cuda/benchmark_results.json`
  - Contains: per-volume times, speedups, GPU name, aggregate stats
- **CSV timing log:** `marching_cubes_cuda.py :: batch_process_volumes() :: lines 259-261`
  - File: `data/out/designB/meshes/marching_cubes_timing.log`
  - Format: `filename\tdevice\tmc_time\ttotal_time\tvertices\tfaces`

**Console output:**

- Per-volume: `benchmark_single_volume()` prints CPU/GPU time, speedup :: lines 68-70, 93-99
- Summary: `benchmark_batch()` prints averages, success rate :: lines 298-308

**Metrics tracked:**

- **Timing:** cpu_time_min, cpu_time_avg, cpu_time_std, gpu_time_min, gpu_time_avg, gpu_time_std
- **Speedup:** cpu_time / gpu_time per volume
- **Throughput:** 1 / gpu_time (volumes/sec)
- **GPU memory:** `torch.cuda.memory_allocated()` :: benchmarks.py line 102

---

### Step 17: Accuracy Metrics (IoU/Chamfer/Hausdorff)

**What happens:** Compare Design B meshes to Design A baseline (quantitative validation)

**Why:** Verify geometric correctness, ensure GPU implementation matches CPU baseline

**How:**

- Load Design A and Design B meshes
- Compute vertex/face count differences
- Approximate Hausdorff distance via sampling (10K points)
- Compute centroid distance, bounding box similarity

**Where:**

- **Verification script:** `designB/python/verify_meshes.py`
  - Mesh comparison: `compare_meshes() :: lines 16-75`
  - Single mesh verification: `verify_single_mesh() :: lines 78-106`
  - Batch verification: `verify_batch() :: lines 109-196`

**Metrics computed:**

- `vertices_diff`, `faces_diff`: absolute count differences
- `vertices_ratio`, `faces_ratio`: Design B / Design A
- `bbox_size_diff`: Euclidean distance between bounding box sizes
- `centroid_distance`: L2 distance between mesh centroids
- `hausdorff_approx`: sample-based Hausdorff distance
- `mean_distance_b_to_a`, `mean_distance_a_to_b`: average point-to-surface distances

**Where metric code:**

- Hausdorff: `verify_meshes.py :: compare_meshes() :: lines 56-70`
  ```python
  from scipy.spatial import cKDTree
  tree_a = cKDTree(samples_a)
  dist_b_to_a, _ = tree_a.query(samples_b)
  metrics['hausdorff_approx'] = float(max(dist_b_to_a.max(), dist_a_to_b.max()))
  ```

**Note:** IoU and Chamfer distance **NOT IMPLEMENTED** (would require voxelization or dense sampling)

**Design C (FaceScape) accuracy metrics:** Planned but not implemented (see Chapter 4.2 section 4.2.5)

---

### Step 18: Results Saving (Logs, CSV, Screenshots, Meshes)

**What happens:** Save all outputs to organized directory structure

**Why:** Reproducibility, thesis figures, performance analysis

**How:**

**Logs:**

- Batch processing log: `data/out/designB/meshes/marching_cubes_timing.log`
  - Written by: `marching_cubes_cuda.py :: batch_process_volumes() :: line 261`
  - Format: TSV with filename, device, mc_time, total_time, vertices, faces

**CSV/JSON:**

- Benchmark results JSON: `data/out/designB/benchmarks_cuda/benchmark_results.json`
  - Written by: `benchmarks.py :: benchmark_batch() :: lines 205-208`
  - Contains: per-volume stats, GPU name, summary aggregates

**Screenshots:** **NOT AUTOMATED**

- Manual: Open meshes in MeshLab, capture screenshots for poster

**Meshes:**

- Directory: `data/out/designB/meshes/`
- Naming: `{volume_stem}.obj` (e.g., `image00002.obj`)
- Format: Wavefront .obj with vertex colors

**Plots:**

- Timing comparison: `benchmarks_cuda/timing_comparison.png`
- Speedup chart: `benchmarks_cuda/speedup_chart.png`
- Generated by: `benchmarks.py :: plot_benchmark_results() :: lines 251-283`

**Where results saved:**

- **Mesh output:** `volume_io.py :: save_mesh_obj() :: line 133`
- **Benchmark JSON:** `benchmarks.py :: line 208` → `json.dump(results_json, f)`
- **Timing log:** `marching_cubes_cuda.py :: line 261` → appends to `.log` file
- **Plots:** `benchmarks.py :: lines 251, 283` → `plt.savefig()`

---

## DESIGN A VS DESIGN B: CODE-LEVEL DIFFERENCES

Based on actual code/config found in the repository:

### 1. Execution Environment

| Aspect             | Design A                                 | Design B                                        |
| ------------------ | ---------------------------------------- | ----------------------------------------------- |
| **Runtime**        | Docker container (asjackson/vrn:latest)  | Native Ubuntu 22.04 + Python venv               |
| **Entry point**    | `docker run ... /runner/run.sh`          | `python3 designB/python/marching_cubes_cuda.py` |
| **VRN inference**  | CPU-only (Torch7, CUDA 7.5 incompatible) | Same (reuses Design A volumes)                  |
| **Marching cubes** | CPU (PyMCubes inside Docker)             | GPU (custom CUDA kernel)                        |

**Code references:**

- Design A: `scripts/batch_process_aflw2000.sh :: line 41` → `docker run ... asjackson/vrn:latest`
- Design B: `scripts/designB_run.sh :: line 85` → `python3 cuda_post/marching_cubes_cuda.py`

---

### 2. Marching Cubes Implementation

| Aspect        | Design A                       | Design B                         |
| ------------- | ------------------------------ | -------------------------------- |
| **Library**   | PyMCubes (C++ wrapper)         | Custom CUDA kernel               |
| **Device**    | CPU single-threaded            | GPU parallel (512 threads/block) |
| **Threshold** | 10.0 on float-cast int8 volume | 0.5 on boolean volume            |
| **Speed**     | ~85.6 ms/volume                | ~4.6 ms/volume (18.36× faster)   |

**Code references:**

- Design A: `raw2obj.py :: line 23` → `mcubes.marching_cubes(vol, 10)`
- Design B: `marching_cubes_cuda.py :: line 54` → `marching_cubes_gpu(volume_tensor, isolevel=0.5)`
- CUDA kernel: `marching_cubes_kernel.cu :: marchingCubesKernel() :: lines 40-141`

---

### 3. Post-Processing Configuration

| Aspect                     | Design A                     | Design B                      |
| -------------------------- | ---------------------------- | ----------------------------- |
| **Axis swap**              | `vertices[:,(2,1,0)]`        | `vertices[:,[2,1,0]]` (same)  |
| **Z-scaling**              | `vertices[:,2] *= 0.5`       | `vertices[:,2] *= 0.5` (same) |
| **Vertex merge tolerance** | ~1.0 (inferred from outputs) | **0.1** (explicit)            |
| **Output vertices**        | ~37,587 avg                  | ~63,571 avg (+69%)            |

**Code references:**

- Design A: `raw2obj.py :: lines 23-24`
- Design B: `volume_io.py :: lines 102-111`
  - **Line 110:** `trimesh.constants.tol.merge = 0.1` ← **KEY QUALITY DIFFERENCE**

---

### 4. Color Mapping

| Aspect                | Design A                   | Design B                          |
| --------------------- | -------------------------- | --------------------------------- |
| **Method**            | Nearest-neighbor (sklearn) | Nearest-neighbor (NumPy indexing) |
| **Image resize**      | 192×192                    | 192×192 (same)                    |
| **Coordinate source** | After transform            | After transform (same)            |

**Code references:**

- Design A: `raw2obj.py :: lines 25-38` (sklearn NearestNeighbors)
- Design B: `volume_io.py :: lines 121-127` (direct NumPy indexing, simpler + faster)

---

### 5. Timing and Benchmarking

| Aspect               | Design A                   | Design B                                   |
| -------------------- | -------------------------- | ------------------------------------------ |
| **Timing tool**      | `/usr/bin/time -v` (shell) | `time.time()` + `torch.cuda.synchronize()` |
| **Runs per volume**  | 1                          | 3 (best-of-3 reported)                     |
| **GPU sync**         | N/A                        | `torch.cuda.synchronize()` before/after    |
| **Benchmark script** | None (manual bash loop)    | `benchmarks.py` (automated)                |

**Code references:**

- Design A: `batch_process_aflw2000.sh :: line 40` → `/usr/bin/time -f ...`
- Design B: `benchmarks.py :: lines 76, 81, 86` → `torch.cuda.synchronize()`

---

### 6. Volume Format

| Aspect      | Design A                          | Design B                          |
| ----------- | --------------------------------- | --------------------------------- |
| **Format**  | .raw binary (int8)                | .npy NumPy array (bool)           |
| **Loading** | `np.fromfile(..., dtype=np.int8)` | `np.load(...)` + `.astype(bool)`  |
| **Storage** | Implicit (Docker internal)        | Explicit (43 .npy files, ~630 MB) |

**Code references:**

- Design A: `raw2obj.py :: line 18` → `vol = np.fromfile(args.volume, dtype=np.int8)`
- Design B: `volume_io.py :: load_volume_npy() :: lines 56-67`

---

### 7. Build Process

| Aspect              | Design A                   | Design B                               |
| ------------------- | -------------------------- | -------------------------------------- |
| **Build required**  | No (prebuilt Docker image) | Yes (CUDA extension compilation)       |
| **Compile command** | N/A                        | `python3 setup.py build_ext --inplace` |
| **Dependencies**    | Docker only                | CUDA 11.8, PyTorch 2.1.0, GCC 9+       |
| **Target SM**       | N/A                        | SM 8.6 (forward-compatible to 8.9)     |

**Code references:**

- Design B build: `designB/setup.py :: lines 1-62`
- CUDA flags: `setup.py :: extra_compile_args :: lines 34-43`

---

## SUMMARY: CRITICAL PATH FOR THESIS

**Design B inference pipeline (GPU-accelerated):**

1. **Load volume:** `volume_io.py :: load_volume_npy()` → 200×192×192 bool array
2. **Transfer to GPU:** `volume_io.py :: volume_to_tensor()` → torch.Tensor on CUDA
3. **CUDA marching cubes:** `cuda_marching_cubes.py :: marching_cubes_gpu()` → raw mesh (172K verts)
4. **Transform coords:** `volume_io.py :: save_mesh_obj()` → axis swap + Z-scale
5. **Merge vertices:** `trimesh.merge_vertices(tol=0.1)` → final mesh (63K verts)
6. **Map colors:** Nearest-neighbor RGB from input image
7. **Export:** Write .obj file

**Benchmarking (for 18.36× speedup claim):**

1. **Run:** `benchmarks.py :: benchmark_batch()` with 43 volumes, 3 runs each
2. **Time CPU:** scikit-image marching cubes (no synchronization needed)
3. **Time GPU:** Custom CUDA kernel with `torch.cuda.synchronize()` before/after
4. **Compute speedup:** cpu_time_min / gpu_time_min per volume
5. **Save results:** JSON with per-volume stats + aggregate mean/std/min/max

**Critical code locations:**

- Main GPU kernel: `marching_cubes_kernel.cu :: marchingCubesKernel() :: lines 40-141`
- Synchronization: `benchmarks.py :: lines 76, 81, 86`
- Post-processing: `volume_io.py :: save_mesh_obj() :: lines 99-131`
- Vertex merge tolerance: `volume_io.py :: line 110` → `0.1` (vs Design A's `1.0`)

---

**End of Code Map**
