# Chapter 4: Design and Implementation (continued)

## 4.2 Preliminary Design and Model Specification

This chapter presents the engineering specification and preliminary design for the VRN 3D face reconstruction pipeline implementations. Building upon the design methodology and requirements established in Section 4.1, this section provides a detailed technical specification of the system architecture, module decomposition, interface definitions, and verification protocols. The specification follows standard systems engineering practices, defining what each design alternative is, how it operates, what guarantees it provides, and how functional correctness will be validated.

---

### 4.2.1 System Overview and Architecture

#### System Boundary and Scope

The VRN 3D face reconstruction system is a multi-stage pipeline that transforms 2D face images into 3D mesh representations. The system boundary encompasses:

**Inputs:**

- 2D face images (JPEG format, variable resolution)
- Dataset metadata (annotations for AFLW2000-3D and FaceScape)
- Configuration parameters (thresholds, device selection, output formats)

**Outputs:**

- 3D mesh files (.obj format with vertex coordinates, face connectivity, and RGB color attributes)
- Intermediate volume files (.npy format, optional for analysis)
- Processing logs and performance metrics
- Visualization artifacts (optional rendered images)

**Processing Stages:**

The end-to-end pipeline consists of five principal stages, represented as a sequential processing flow:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Face Detection and Preprocessing                      │
│ - Input: Raw 2D image (JPEG)                                   │
│ - Process: dlib face detector → 68 landmark detection →        │
│            alignment based on eye positions → crop to 450×450   │
│ - Output: Normalized face crop (.crop.jpg)                     │
│ - Failure Mode: No face detected → pipeline terminates         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Volumetric Regression (VRN Inference)                 │
│ - Input: Aligned face crop (450×450×3 RGB)                     │
│ - Process: Torch7 CNN forward pass → 3D convolutional layers   │
│ - Output: Volumetric prediction (200×192×192 voxel grid)       │
│ - Properties: Continuous occupancy values [0, 1]               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Volume Export and Format Conversion                   │
│ - Input: VRN volume tensor (200×192×192 float32)               │
│ - Process: Serialize to disk format (.raw → .npy conversion)   │
│ - Output: NumPy array file (.npy) for downstream processing    │
│ - Design A: Implicit (in-memory) | Design B: Explicit export   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Isosurface Extraction (Marching Cubes)                │
│ - Input: Voxel volume (200×192×192), threshold T               │
│ - Process: Parallel marching cubes → triangle mesh generation  │
│ - Output: Raw mesh (vertices: Nx3 float, faces: Mx3 int)       │
│ - Implementation: Design A (CPU) | Design B (CUDA GPU)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Mesh Post-Processing and Export                       │
│ - Input: Raw marching cubes mesh                               │
│ - Process: Axis permutation (Z,Y,X → X,Y,Z) → Z-scaling (×0.5) │
│            → Vertex merging (tolerance 0.1) → RGB color mapping │
│ - Output: Final mesh (.obj format, post-processed)             │
│ - Properties: Manifold surface, correct orientation, colors    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Characteristics:**

1. **Modularity**: Each stage has well-defined inputs and outputs, enabling independent testing and validation.

2. **Failure Propagation**: Failures in Stage 1 (face detection) terminate the pipeline immediately; failures in later stages are logged but do not affect upstream processing.

3. **Intermediate Persistence**: Design B explicitly exports volumes between stages, enabling reprocessing without re-running expensive VRN inference.

4. **Acceleration Points**: Stage 4 (marching cubes) is the primary target for GPU acceleration; other stages remain CPU-bound due to legacy constraints.

5. **Determinism**: All operations are deterministic given identical inputs, ensuring reproducible outputs for validation and comparison.

---

### 4.2.2 Requirements Traceability

This section establishes traceability from the high-level requirements defined in Section 4.1.2 to the design specifications presented in this chapter.

#### Functional Requirements Traceability

| Requirement ID | Requirement Description                        | Design Satisfaction                                                  | Verification Method                               |
| -------------- | ---------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| **FR1**        | End-to-end mesh generation (.obj format)       | All designs generate .obj files with vertices, faces, and RGB colors | File format validation, mesh viewer compatibility |
| **FR2**        | Dataset compatibility (AFLW2000-3D, FaceScape) | Design A/B: AFLW2000; Design C: FaceScape support                    | Dataset loader testing, preprocessing validation  |
| **FR3**        | Format interoperability (MeshLab, Blender)     | .obj format follows Wavefront standard                               | Load test in multiple viewers                     |
| **FR4**        | Batch processing support                       | Scripts support multiple images with logging                         | Batch execution test (43+ images)                 |
| **FR5**        | Hardware portability                           | Design A: CPU-only; Design B: GPU required; both on Ubuntu 22.04     | Cross-platform testing                            |

#### Performance Requirements Traceability

| Requirement ID | Requirement Description                 | Design Satisfaction                        | Verification Method              |
| -------------- | --------------------------------------- | ------------------------------------------ | -------------------------------- |
| **PR1**        | ≥10× speedup on marching cubes          | Design B achieves 18.36× average speedup   | Benchmark timing measurements    |
| **PR2**        | ≥50 volumes/sec throughput              | Design B GPU: 217 volumes/sec              | Throughput profiling             |
| **PR3**        | Latency consistency (std dev <10% mean) | Design B GPU: σ=0.2ms (4.3% of 4.6ms mean) | Statistical timing analysis      |
| **PR4**        | GPU memory <100 MB per volume           | Design B: ~28 MB allocated per volume      | GPU memory profiling             |
| **PR5**        | Statistical benchmarking rigor          | 3 runs per volume, warm-up iterations      | Documented benchmarking protocol |

#### Quality Requirements Traceability

| Requirement ID | Requirement Description                             | Design Satisfaction                                        | Verification Method                            |
| -------------- | --------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| **QR1**        | Geometric correctness (axis scaling, coordinates)   | Post-processing includes Z-scaling, axis permutation       | Visual inspection, coordinate range validation |
| **QR2**        | Topological consistency (manifold surfaces)         | Vertex merging produces clean topology                     | Mesh analysis (trimesh validation)             |
| **QR3**        | Visual plausibility (smooth surfaces, no artifacts) | Higher vertex counts (63,571 vs 37,587) improve smoothness | MeshLab visual inspection                      |
| **QR4**        | Color fidelity (RGB mapping from input images)      | RGB mapping performed after spatial transformation         | Color distribution analysis                    |
| **QR5**        | Quantitative proxy metrics                          | Vertex counts, extraction consistency measured             | Comparative analysis Design A vs B             |

**Traceability Summary**: All functional, performance, and quality requirements from Section 4.1.2 are satisfied by at least one design alternative, with verification methods specified in Section 4.2.8.

---

### 4.2.3 Design A Specification: Legacy CPU Baseline

#### Overview

Design A implements the original VRN pipeline as a CPU-only Docker containerized application. This design serves as the correctness baseline and performance reference for comparison with accelerated alternatives.

#### Runtime Environment

**Container**: `asjackson/vrn:latest`

- **Base OS**: Ubuntu 16.04 (legacy)
- **Framework**: Torch7 (2016 release)
- **CUDA Support**: CUDA 7.5 / cuDNN 5.1 (incompatible with modern GPUs, runs CPU-only)
- **Dependencies**: dlib 19.4, OpenCV 3.1, Boost 1.58

**Host Requirements**:

- **OS**: Ubuntu 22.04 LTS (tested)
- **Docker**: 24.0.5 or newer
- **Memory**: 4 GB RAM minimum
- **Storage**: 10 GB for container + dataset + outputs

#### Pipeline Specification

**Stage 1: Face Detection**

- **Algorithm**: dlib HOG-based face detector
- **Input**: JPEG image (variable resolution)
- **Output**: Face bounding box coordinates, 68 facial landmarks
- **Failure Condition**: No face detected → logs error, skips mesh generation
- **Success Rate**: 86% (43 of 50 AFLW2000 subset images)

**Stage 2: Preprocessing**

- **Alignment**: Similarity transform based on eye positions
- **Cropping**: Center and scale face to 450×450 pixels
- **Normalization**: RGB values [0, 255] → [0, 1] float32
- **Output**: `.crop.jpg` file (450×450×3 JPEG)

**Stage 3: VRN Inference**

- **Model**: vrn-unguided.t7 (Torch7 serialized checkpoint)
- **Architecture**: Stacked hourglass CNN with 3D convolutions
- **Input**: 450×450×3 RGB tensor
- **Processing**: ~2–3 seconds per image (CPU, single-threaded)
- **Output**: 200×192×192 voxel volume (occupancy probabilities)

**Stage 4: Marching Cubes (CPU)**

- **Implementation**: PyMCubes (Python binding to C++ marching cubes)
- **Input**: 200×192×192 boolean volume (threshold T=0.5)
- **Processing**: ~85.6ms per volume (single-threaded CPU)
- **Output**: Raw mesh (vertices: Nx3 float, faces: Mx3 int)
- **Typical Size**: ~172,000 vertices (pre-merge), ~140,000 faces

**Stage 5: Post-Processing**

- **Axis Permutation**: Z, Y, X → X, Y, Z (correct spatial orientation)
- **Z-Scaling**: `vertices[:, 2] *= 0.5` (compress depth dimension)
- **Vertex Merging**: Merge duplicate vertices (tolerance=1.0)
- **RGB Mapping**: Map vertex coordinates to input image pixels for colors
- **Output**: Final mesh (~37,587 vertices post-merge, ~125,290 faces)

**Stage 6: Export**

- **Format**: Wavefront .obj (ASCII)
- **Fields**: Vertex positions (v x y z), face connectivity (f i j k), vertex colors (optional)
- **File Size**: ~3.5 MB per mesh

#### Input and Output Specification

**Input Dataset**: AFLW2000-3D subset

- **Subset Size**: 50 images (documented in `docs/aflw2000_subset.txt`)
- **Image Format**: JPEG, variable resolution (typically 450×450 after cropping)
- **Annotations**: 68 3D facial landmarks (not used by VRN, only for evaluation)

**Output Files** (per input image `imageXXXXX.jpg`):

- `imageXXXXX.jpg.obj` - 3D mesh file
- `imageXXXXX.jpg.crop.jpg` - Aligned face crop
- `batch_process.log` - Execution log with timing and status
- `time.log` - Per-image timing breakdown

**Directory Structure**:

```
data/
├── in/
│   └── aflw2000/          # Input images (50 JPEG files)
├── out/
│   └── designA/           # Design A outputs
│       ├── *.obj          # Mesh files (43 successful)
│       ├── *.crop.jpg     # Face crops (43 successful)
│       ├── batch_process.log
│       └── time.log
└── tmp/                   # Temporary scratch space
```

#### Reproducibility Instructions

**Execution Command** (single image):

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  asjackson/vrn:latest \
  /runner/run.sh /data/in/aflw2000/image00002.jpg
```

**Batch Processing**:

```bash
./scripts/batch_process_aflw2000.sh
```

**Expected Behavior**:

- **Successful Images**: Generate .obj and .crop.jpg in 11–18 seconds
- **Failed Images**: Log "No face detected", complete in 1–2 seconds
- **Output Validation**: Mesh files should load in MeshLab without errors

#### Performance Baseline

- **Average Processing Time**: 10.26 seconds/image (includes failures)
- **Successful Reconstruction Time**: 11–18 seconds/image
- **Marching Cubes Component**: ~85.6ms (3.3% of total time)
- **Throughput**: ~5.9 images/minute
- **Success Rate**: 86% (43/50)

#### Failure Modes

1. **Face Detection Failure**: Extreme poses (yaw >70°), occlusion, low resolution
2. **Docker Runtime Error**: Insufficient memory, permission issues
3. **Model Loading Failure**: Corrupted checkpoint file (rare)

**Note**: Design A does not experience failures after successful face detection; VRN inference and mesh extraction are deterministic and robust.

---

### 4.2.4 Design B Specification: Hybrid CPU Inference + CUDA Mesh Extraction

#### Overview

Design B implements a two-stage hybrid pipeline that accelerates mesh extraction via custom CUDA kernels while maintaining VRN inference in the legacy Docker environment. This design demonstrates measurable GPU acceleration without requiring full pipeline modernization.

#### Architecture: Two-Stage Pipeline

**Stage 1: VRN Volume Extraction (Legacy CPU)**

- **Environment**: Same as Design A (Docker, Torch7, CPU-only)
- **Process**: Face detection → VRN inference → volume export
- **Output**: 200×192×192 voxel volume (.raw or .npy format)
- **Modification**: Process.lua script modified to export raw volumes instead of generating meshes

**Stage 2: CUDA Marching Cubes (Modern GPU)**

- **Environment**: Ubuntu 22.04, Python 3.10, PyTorch 2.1.0, CUDA 11.8
- **Process**: Load volume → GPU marching cubes → post-processing → mesh export
- **Output**: Final .obj mesh file (post-processed)

#### Stage 1 Specification: Volume Export

**Input**: AFLW2000-3D images (same as Design A)

**Processing**:

1. Face detection and cropping (same as Design A)
2. VRN inference (same as Design A)
3. Volume extraction from Torch7 tensor
4. Serialization to .raw format (binary uint8)
5. Conversion to .npy format (NumPy float32)

**Output Volume Properties**:

- **Shape**: 200×192×192 voxels
- **Dtype**: float32 (normalized [0, 1] occupancy probabilities)
- **File Format**: NumPy .npy (binary serialization)
- **File Size**: ~14.7 MB per volume (200×192×192×4 bytes)
- **Storage**: `data/out/volumes/*.npy`

**Thresholding Assumptions**:

- Volume values represent occupancy probabilities
- Boolean thresholding applied at T=0.5 before marching cubes
- Expected voxel occupancy: ~7.8% (after thresholding)

#### Stage 2 Specification: CUDA Marching Cubes

**Module**: Custom CUDA kernel implemented as PyTorch C++ extension

**Input**:

- Volume tensor: `torch.Tensor` of shape (200, 192, 192), dtype `float32`, device `cuda`
- Threshold: float (default T=0.5)
- Output buffers: Pre-allocated tensors for vertices and faces

**CUDA Kernel Configuration**:

- **Thread Block Dimensions**: 8×8×8 threads (512 threads per block)
- **Grid Dimensions**: Calculated as `ceil(dim / 8)` for each axis
  - Example: For 200×192×192 volume → grid (25, 24, 24)
- **Total Threads**: ~14,400 (matching voxel grid size)
- **Memory Layout**: Assumes contiguous memory (C-order: Z, Y, X indexing)

**Algorithm**:

1. **Voxel Processing** (per-thread):
   - Read 8 corner values for current voxel cube
   - Compute cube index (0-255) based on threshold
   - Lookup edge table to determine active edges
   - Interpolate vertex positions on active edges
   - Lookup triangle table to generate triangles (0-5 triangles per cube)

2. **Output Buffering**:
   - Atomic counters for vertex and face indices
   - Dynamic allocation with pre-sized buffers (conservative estimate)
   - Deduplication performed in post-processing

**CUDA Extension Design**:

**Binding Layer** (`marching_cubes_bindings.cpp`):

- **Framework**: PyBind11 C++17 bindings
- **Functions**:
  - `marching_cubes_forward(volume, threshold, device_id)`
  - `check_cuda_available()`
- **Tensor Validation**: Check device placement, dtype, memory layout
- **Stream Management**: Uses default CUDA stream for simplicity

**Build Configuration** (`setup.py`):

- **Compilation Target**: SM 8.6 (Ampere, maximum supported by CUDA 11.8)
- **Forward Compatibility**: Runs on SM 8.9 (Ada Lovelace) via binary compatibility
- **Compiler Flags**:
  - `-O3` (maximum optimization)
  - `--use_fast_math` (GPU math approximations)
  - `-std=c++17` (required by PyTorch 2.x)
  - `-arch=sm_86` (compute capability)

**Output**:

- **Vertices**: Nx3 float32 tensor (N typically 145,000–220,000 pre-merge)
- **Faces**: Mx3 int32 tensor (M typically 48,000–73,000 pre-merge)
- **Processing Time**: ~4.6ms per volume (GPU), ~84.2ms (CPU baseline)

#### Post-Processing Specification

**Inputs**:

- Raw marching cubes mesh (vertices, faces)
- Input volume (for RGB color mapping)
- Configuration: merge tolerance, scaling factors

**Transformations** (applied sequentially):

1. **Axis Permutation**:
   - Input order: Z, Y, X (VRN volume indexing)
   - Output order: X, Y, Z (standard mesh coordinates)
   - Implementation: `vertices = vertices[:, [2, 1, 0]]`

2. **Z-Axis Scaling**:
   - Compress depth dimension by factor of 0.5
   - Implementation: `vertices[:, 2] *= 0.5`
   - Rationale: Corrects for aspect ratio distortion in VRN volumes

3. **Vertex Merging**:
   - Algorithm: Octree-based duplicate detection (trimesh library)
   - Tolerance: 0.1 (10× smaller than Design A's 1.0)
   - Effect: Preserves fine details, increases vertex count
   - Output: Reduces ~172,000 raw vertices → ~63,571 post-merge

4. **RGB Color Mapping**:
   - Map vertex coordinates back to input image pixels
   - Nearest-neighbor interpolation
   - Performed _after_ spatial transformations to ensure correct vertex-color correspondence
   - Implementation: Index volume with transformed coordinates

**Output**:

- **Post-Processed Mesh**: ~63,571 vertices, ~211,903 faces (average)
- **File Format**: Wavefront .obj (same as Design A)
- **File Size**: ~5.9 MB (larger due to higher vertex count)

#### Interface Specification: Design B Pipeline

**Python API**:

```python
from cuda_post.marching_cubes_cuda import marching_cubes_gpu_pytorch

# Load volume
volume = np.load('data/out/volumes/image00002.npy')
volume_tensor = torch.from_numpy(volume).float().cuda()

# Extract mesh
vertices, faces = marching_cubes_gpu_pytorch(volume_tensor, threshold=0.5)

# Post-process
vertices = transform_and_merge(vertices, tolerance=0.1)
colors = map_rgb_colors(vertices, input_image)

# Export
export_obj(vertices, faces, colors, 'output.obj')
```

**Configuration Parameters**:

- `threshold`: float (default 0.5, range [0, 1])
- `merge_tolerance`: float (default 0.1, smaller = more vertices)
- `device`: str (`'cuda'` or `'cpu'`)
- `block_size`: tuple (default (8, 8, 8), must match kernel compilation)

#### Build and Deployment Specification

**Build Requirements**:

- **Python**: 3.10+ with pip
- **PyTorch**: 2.1.0 (cu118 variant)
- **CUDA Toolkit**: 11.8 or 12.x
- **Compiler**: GCC 9+, nvcc (from CUDA toolkit)
- **Build Tools**: setuptools, ninja (optional but recommended)

**Build Instructions**:

```bash
# 1. Activate environment
source vrn_env/bin/activate

# 2. Install dependencies
pip install torch==2.1.0+cu118 numpy scikit-image trimesh

# 3. Compile CUDA extension
cd designB/cuda_kernels
python setup.py build_ext --inplace

# 4. Verify installation
python -c "import marching_cubes_cuda_ext; print('OK')"
```

**Build Artifacts**:

- `marching_cubes_cuda_ext.cpython-310-x86_64-linux-gnu.so` (shared library)
- Build logs: `build/*.log`

**Deployment**:

- Extension must be in Python path or same directory as calling script
- CUDA runtime libraries must be accessible (LD_LIBRARY_PATH)
- GPU driver must support compute capability 8.6+ (tested on RTX 4070 SUPER)

#### Performance Guarantees

- **Speedup**: ≥16× over CPU baseline (measured: 18.36× average, range 16.51×–19.58×)
- **Throughput**: ≥50 volumes/sec (measured: 217 volumes/sec)
- **Latency**: <10ms per volume (measured: 4.6ms average)
- **Memory**: <100 MB GPU allocation per volume (measured: ~28 MB)
- **Consistency**: Standard deviation <10% of mean (measured: 4.3%)

#### Failure Modes

1. **CUDA Extension Build Failure**: Incompatible CUDA version, missing compiler
2. **GPU Memory Overflow**: Volume too large, insufficient VRAM
3. **Kernel Launch Failure**: Invalid grid/block configuration, device unavailable
4. **Post-Processing Error**: Invalid mesh topology (rare, caught by trimesh validation)

**Error Handling**:

- All CUDA errors logged with context
- CPU fallback available (automatically used if GPU unavailable)
- Invalid meshes logged but do not crash pipeline

---

### 4.2.5 Design C Preliminary Specification: FaceScape Dataset Adaptation (Planned)

#### Overview

Design C extends Design B to process the FaceScape dataset, enabling evaluation of VRN generalization to high-quality studio-captured 3D face scans. This design is **planned but not yet implemented**; this section provides a preliminary specification to guide future development.

#### Objectives

1. **Domain Shift Evaluation**: Assess VRN performance on FaceScape (controlled lighting, high resolution, diverse expressions) vs. AFLW2000 (in-the-wild, varied quality)
2. **Quantitative Validation**: Leverage FaceScape ground-truth 3D scans for geometric error metrics (Chamfer distance, Hausdorff distance, normal consistency)
3. **Expression Generalization**: Evaluate reconstruction quality across neutral and expressive faces

#### Dataset Specification: FaceScape

**Source**: FaceScape Benchmark (Chen et al., 2020)

- **Subjects**: 847 subjects, 16,940 scans
- **Capture Setup**: 68 DSLR cameras, studio lighting, structured light scanning
- **Resolution**: High-quality 3D meshes (~30,000–100,000 vertices per scan)
- **Expressions**: Neutral + 20 expression categories
- **Annotations**: JSON metadata (subject ID, expression, camera calibration)

**Planned Subset Selection**:

- **Size**: 100 subjects × 3 expressions = 300 scans
- **Split**: Training views (not used), evaluation views (frontal camera angles)
- **Criteria**: Select frontal views matching AFLW2000 pose distribution

#### Pipeline Modifications

**Stage 1: Data Ingestion**

- **Input Format**: FaceScape provides RGB images + 3D mesh pairs
- **Preprocessing**: Extract frontal view images, resize to match VRN input (450×450)
- **Alignment**: Use FaceScape landmarks (provided) instead of dlib detection
- **Expected Change**: Face detection stage replaced by metadata-based cropping

**Stage 2: VRN Inference**

- **No Change**: VRN model and inference remain identical to Design B
- **Volume Export**: Same 200×192×192 .npy format

**Stage 3: Marching Cubes**

- **No Change**: Same CUDA kernel as Design B
- **Threshold**: May require per-dataset tuning (investigate T=0.4–0.6 range)

**Stage 4: Post-Processing**

- **Coordinate Alignment**: May require different axis permutation or scaling to match FaceScape coordinate frame
- **Mesh Registration**: Procrustes alignment to ground truth (for error computation)

**Stage 5: Evaluation**

- **New Module**: Quantitative error metrics
  - Chamfer distance: Average nearest-neighbor distance between predicted and GT meshes
  - Hausdorff distance: Maximum nearest-neighbor distance (captures outliers)
  - Normal consistency: Cosine similarity of vertex normals
- **Visualization**: Overlay prediction on ground truth, color-code error heatmap

#### Interface Changes

**New Data Loader**:

```python
class FaceScapeDataset:
    def __init__(self, root_dir, split='eval'):
        # Load JSON metadata
        # Index subjects, expressions, camera views
        pass

    def __getitem__(self, idx):
        # Return: RGB image, GT mesh path, metadata
        pass
```

**New Evaluation Metrics**:

```python
def compute_chamfer_distance(pred_mesh, gt_mesh):
    # Nearest neighbor search, bidirectional distance
    pass

def compute_hausdorff_distance(pred_mesh, gt_mesh):
    # Maximum point-to-surface distance
    pass
```

#### Expected Outputs

**Per-Sample Outputs**:

- Predicted mesh (.obj)
- Ground truth mesh (.obj) from FaceScape
- Error metrics (JSON): `{"chamfer": 2.3, "hausdorff": 8.7, "normal_cos": 0.92}`
- Error heatmap visualization (.png)

**Aggregate Metrics** (across 300 samples):

- Mean ± std Chamfer distance
- Median Hausdorff distance
- Expression-wise breakdown (neutral vs. smiling vs. frowning, etc.)

#### Risks and Unknowns

1. **Domain Shift**: VRN trained on 300W-LP (in-the-wild) may not generalize to FaceScape (studio)
   - **Mitigation**: Qualitative analysis of failure modes, potential retraining

2. **Coordinate Frame Mismatch**: FaceScape and VRN use different coordinate systems
   - **Mitigation**: Implement manual alignment with rotation/translation estimation

3. **Mesh Resolution Disparity**: FaceScape GT meshes are very high-resolution (100K vertices), VRN outputs ~63K
   - **Mitigation**: Downsample GT meshes or use point-to-surface metrics instead of vertex-to-vertex

4. **Annotation Availability**: FaceScape landmarks may not match dlib 68-point format
   - **Mitigation**: Write custom alignment code or use facial keypoints from FaceScape metadata

5. **Computational Cost**: Processing 300 scans + error computation increases runtime
   - **Mitigation**: Leverage Design B's GPU acceleration; parallelize evaluation metrics

#### Verification Plan (Preliminary)

**Unit Tests**:

- FaceScape data loader: Load sample, verify image dimensions, check GT mesh validity
- Coordinate alignment: Test known transformation matrices
- Error metrics: Validate against synthetic meshes (sphere, cube)

**Integration Tests**:

- End-to-end run on 5 FaceScape samples
- Verify predicted meshes load in MeshLab
- Check error metrics are in expected range (e.g., Chamfer <5mm for good predictions)

**Acceptance Criteria**:

- At least 90% of samples produce valid meshes (non-empty, manifold)
- Chamfer distance correlates with visual quality (manual inspection of best/worst 10 samples)
- Error metrics are reproducible across runs

---

### 4.2.6 Module Decomposition and Responsibilities

This section decomposes the VRN pipeline into discrete modules, specifying responsibilities, inputs, outputs, dependencies, and failure modes for each component.

#### Module Inventory

| Module ID | Module Name                      | Primary Responsibility                                |
| --------- | -------------------------------- | ----------------------------------------------------- |
| **M1**    | Data Ingestion & Dataset Manager | Load images, parse annotations, manage dataset splits |
| **M2**    | Face Detection & Preprocessing   | Detect face, extract landmarks, align, crop           |
| **M3**    | VRN Inference Engine             | Volumetric regression via Torch7 CNN                  |
| **M4**    | Volume Exporter & Format Adapter | Serialize volumes to disk (.raw, .npy)                |
| **M5**    | Marching Cubes (CPU)             | Extract isosurface using scikit-image                 |
| **M6**    | Marching Cubes (CUDA)            | Extract isosurface using custom CUDA kernel           |
| **M7**    | Mesh Post-Processor              | Transform, merge, color map meshes                    |
| **M8**    | Mesh Exporter                    | Write .obj files, validate format                     |
| **M9**    | Benchmark & Profiler             | Measure timing, GPU memory, throughput                |
| **M10**   | Visualization & Analysis         | Render meshes, generate plots, error heatmaps         |

#### Detailed Module Specifications

**M1: Data Ingestion & Dataset Manager**

- **Responsibilities**:
  - Load image files from disk (JPEG, PNG)
  - Parse dataset annotations (MATLAB .mat for AFLW2000, JSON for FaceScape)
  - Provide iterator interface for batch processing
  - Track processing status (success/failure per sample)

- **Inputs**:
  - Dataset root directory path
  - Subset file list (e.g., `aflw2000_subset.txt`)
  - Configuration (train/val/test split if applicable)

- **Outputs**:
  - Image data (NumPy array or PIL Image)
  - Metadata (filename, subject ID, annotations)
  - Sample index for logging

- **Dependencies**: NumPy, Pillow, scipy (for .mat files)

- **Failure Modes**:
  - File not found (missing image)
  - Corrupted image file
  - Invalid annotation format

**M2: Face Detection & Preprocessing**

- **Responsibilities**:
  - Detect face bounding box using dlib
  - Extract 68 facial landmarks
  - Compute alignment transformation (similarity transform based on eyes)
  - Crop and resize to 450×450 pixels

- **Inputs**:
  - RGB image (variable resolution)

- **Outputs**:
  - Aligned face crop (450×450×3 NumPy array)
  - Transformation matrix (for inverse mapping if needed)
  - Success flag (True if face detected, False otherwise)

- **Dependencies**: dlib, OpenCV

- **Failure Modes**:
  - No face detected (extreme pose, occlusion)
  - Multiple faces detected (ambiguity)
  - Landmark detection failure

**M3: VRN Inference Engine**

- **Responsibilities**:
  - Load pretrained VRN model (vrn-unguided.t7)
  - Perform forward pass: 450×450×3 → 200×192×192 volume
  - Normalize input and denormalize output

- **Inputs**:
  - Aligned face crop (450×450×3 float32, [0, 1] normalized)

- **Outputs**:
  - Volumetric prediction (200×192×192 float32, occupancy probabilities)

- **Dependencies**: Torch7, Lua runtime

- **Failure Modes**:
  - Model file missing or corrupted
  - Out-of-memory (unlikely with 200×192×192 output)
  - Torch7 runtime errors (rare)

**M4: Volume Exporter & Format Adapter**

- **Responsibilities**:
  - Convert Torch7 tensor to NumPy array
  - Serialize volume to disk (.raw binary, .npy format)
  - Ensure correct byte order and dtype

- **Inputs**:
  - VRN output volume (Torch7 tensor)

- **Outputs**:
  - Volume file (.npy, 200×192×192 float32)

- **Dependencies**: NumPy, struct (for .raw binary)

- **Failure Modes**:
  - Disk write error (insufficient space, permissions)
  - Dtype conversion error

**M5: Marching Cubes (CPU)**

- **Responsibilities**:
  - Apply boolean thresholding to volume
  - Execute marching cubes algorithm (scikit-image)
  - Generate raw mesh (vertices, faces)

- **Inputs**:
  - Volume (200×192×192 float32 or bool)
  - Threshold T (float, default 0.5)

- **Outputs**:
  - Vertices (Nx3 float64)
  - Faces (Mx3 int32)
  - Normals (Nx3 float64, optional)

- **Dependencies**: scikit-image, NumPy

- **Failure Modes**:
  - Empty volume (all zeros) → empty mesh
  - Invalid threshold (T<0 or T>1)

**M6: Marching Cubes (CUDA)**

- **Responsibilities**:
  - Transfer volume to GPU memory
  - Launch CUDA kernel with 8×8×8 thread blocks
  - Retrieve vertices and faces from GPU
  - Handle CUDA errors and memory allocation

- **Inputs**:
  - Volume tensor (200×192×192 float32, device='cuda')
  - Threshold T (float)
  - Device ID (int, default 0)

- **Outputs**:
  - Vertices (Nx3 float32, device='cuda' or 'cpu')
  - Faces (Mx3 int32, device='cuda' or 'cpu')

- **Dependencies**: PyTorch 2.1.0+cu118, custom CUDA extension

- **Failure Modes**:
  - CUDA out of memory (volume too large)
  - Kernel launch failure (invalid grid/block config)
  - GPU not available (fallback to CPU)

**M7: Mesh Post-Processor**

- **Responsibilities**:
  - Axis permutation (Z,Y,X → X,Y,Z)
  - Z-axis scaling (×0.5)
  - Vertex deduplication and merging (trimesh)
  - RGB color mapping from input image

- **Inputs**:
  - Raw mesh (vertices, faces)
  - Input image (for color mapping)
  - Configuration (merge tolerance, scaling factors)

- **Outputs**:
  - Post-processed mesh (vertices, faces, colors)

- **Dependencies**: trimesh, NumPy, Pillow

- **Failure Modes**:
  - Invalid mesh topology (non-manifold)
  - Color mapping out of bounds

**M8: Mesh Exporter**

- **Responsibilities**:
  - Format mesh data as Wavefront .obj
  - Write vertex positions, face connectivity, vertex colors
  - Validate output file format

- **Inputs**:
  - Mesh (vertices, faces, colors)
  - Output file path

- **Outputs**:
  - .obj file (ASCII text)

- **Dependencies**: None (standard library file I/O)

- **Failure Modes**:
  - Disk write error
  - Invalid vertex/face indices

**M9: Benchmark & Profiler**

- **Responsibilities**:
  - Measure execution time per pipeline stage
  - Profile GPU memory usage (PyTorch profiler)
  - Record throughput (volumes/sec)
  - Generate timing statistics (mean, std, min, max)

- **Inputs**:
  - Pipeline execution context

- **Outputs**:
  - Timing logs (JSON, CSV)
  - Profiling reports (text, plots)

- **Dependencies**: time, torch.cuda (for GPU profiling)

- **Failure Modes**:
  - Clock synchronization issues (use `torch.cuda.synchronize()`)

**M10: Visualization & Analysis**

- **Responsibilities**:
  - Render meshes in MeshLab or Matplotlib
  - Generate comparison plots (CPU vs GPU timing)
  - Create error heatmaps (Design C)
  - Export visualizations (PNG, MP4)

- **Inputs**:
  - Mesh files
  - Timing data
  - Error metrics (Design C)

- **Outputs**:
  - Rendered images (.png)
  - Plots (.png, .pdf)
  - Videos (.mp4, optional)

- **Dependencies**: MeshLab (external), Matplotlib, Open3D (optional)

- **Failure Modes**:
  - MeshLab not installed
  - Invalid mesh file format

---

### 4.2.7 Interface and Data Specification

This section defines the data formats, tensor shapes, and configuration interfaces used throughout the pipeline.

#### Volume Specification

| Property               | Value                         | Notes                                         |
| ---------------------- | ----------------------------- | --------------------------------------------- |
| **Shape**              | (200, 192, 192)               | Z × Y × X indexing (NumPy/PyTorch convention) |
| **Dtype**              | float32                       | Occupancy probabilities [0, 1]                |
| **File Format**        | .npy (NumPy serialized array) | Binary, includes dtype and shape metadata     |
| **Size**               | 14.7 MB                       | 200 × 192 × 192 × 4 bytes                     |
| **Threshold**          | 0.5 (default)                 | Boolean thresholding for marching cubes       |
| **Expected Occupancy** | ~7.8% (after thresholding)    | Typical for face volumes                      |

**Example Access Pattern**:

```python
volume = np.load('image00002.npy')  # Shape: (200, 192, 192)
binary_volume = (volume > 0.5).astype(bool)
```

#### Mesh Specification

**Raw Marching Cubes Output** (pre-merge):

| Property      | Design A (CPU) | Design B (GPU) | Notes                           |
| ------------- | -------------- | -------------- | ------------------------------- |
| **Vertices**  | Nx3 float64    | Nx3 float32    | N ≈ 145,000–220,000 (pre-merge) |
| **Faces**     | Mx3 int32      | Mx3 int32      | M ≈ 48,000–73,000 (pre-merge)   |
| **Average N** | ~172,000       | ~172,000       | Similar pre-merge counts        |

**Post-Processed Mesh Output**:

| Property             | Design A       | Design B       | Notes                            |
| -------------------- | -------------- | -------------- | -------------------------------- |
| **Vertices**         | ~37,587 (avg)  | ~63,571 (avg)  | After merge (tol=1.0 vs 0.1)     |
| **Faces**            | ~125,290 (avg) | ~211,903 (avg) | Proportional to vertex count     |
| **Vertex Range (X)** | 45–133         | 57.5–139.5     | Spatial extent (arbitrary units) |
| **Vertex Range (Y)** | 33–175         | 53.5–174.5     | Height axis                      |
| **Vertex Range (Z)** | 17–88          | 15.8–88.2      | Depth axis (after Z-scaling)     |
| **Colors**           | RGB per vertex | RGB per vertex | uint8 [0, 255]                   |

**File Format (.obj)**:

```
# Wavefront OBJ format
v x y z r g b    # Vertex position + color
f i j k          # Face (triangle) connectivity (1-indexed)
```

#### Configuration Parameters

| Parameter         | Type  | Default        | Range            | Description                   |
| ----------------- | ----- | -------------- | ---------------- | ----------------------------- |
| `threshold`       | float | 0.5            | [0, 1]           | Marching cubes iso-value      |
| `merge_tolerance` | float | 0.1 (Design B) | [0.01, 1.0]      | Vertex deduplication distance |
| `device`          | str   | 'cuda'         | {'cuda', 'cpu'}  | Computation device            |
| `device_id`       | int   | 0              | [0, N_GPU-1]     | GPU index (multi-GPU systems) |
| `block_size`      | tuple | (8, 8, 8)      | Fixed at compile | CUDA thread block dimensions  |
| `z_scale_factor`  | float | 0.5            | [0.1, 1.0]       | Z-axis compression factor     |

#### Dataset Specification

**AFLW2000-3D Subset**:

| Property               | Value                             |
| ---------------------- | --------------------------------- |
| **Total Images**       | 50                                |
| **Successful Volumes** | 43                                |
| **Failed Images**      | 7 (face detection failures)       |
| **Image Format**       | JPEG, variable resolution         |
| **Crop Size**          | 450×450 pixels                    |
| **Subset File**        | `docs/aflw2000_subset.txt`        |
| **Annotations**        | 68 3D landmarks (not used by VRN) |

**File Naming Convention**:

- Input: `image00002.jpg`, `image00006.jpg`, ...
- Volume: `image00002.npy`, `image00006.npy`, ...
- Mesh: `image00002.obj`, `image00006.obj`, ...

**FaceScape (Design C, Planned)**:

| Property           | Value                                        |
| ------------------ | -------------------------------------------- |
| **Total Scans**    | 16,940 (full dataset)                        |
| **Planned Subset** | 300 (100 subjects × 3 expressions)           |
| **Image Format**   | PNG, high resolution (2K–4K)                 |
| **Ground Truth**   | 3D mesh (.obj, ~30K–100K vertices)           |
| **Annotations**    | JSON (subject ID, expression, camera params) |

#### Interface Summary Table

| Interface       | Producer Module        | Consumer Module        | Format                | Shape/Type                                   |
| --------------- | ---------------------- | ---------------------- | --------------------- | -------------------------------------------- |
| **Input Image** | M1 (Data Ingestion)    | M2 (Face Detection)    | JPEG/PNG              | (H, W, 3) uint8                              |
| **Face Crop**   | M2 (Preprocessing)     | M3 (VRN Inference)     | NumPy array           | (450, 450, 3) float32                        |
| **VRN Volume**  | M3 (VRN Inference)     | M4 (Volume Exporter)   | Torch7 tensor         | (200, 192, 192) float32                      |
| **Volume File** | M4 (Volume Exporter)   | M5/M6 (Marching Cubes) | .npy file             | (200, 192, 192) float32                      |
| **Raw Mesh**    | M5/M6 (Marching Cubes) | M7 (Post-Processor)    | NumPy arrays          | Vertices: (N,3), Faces: (M,3)                |
| **Final Mesh**  | M7 (Post-Processor)    | M8 (Exporter)          | NumPy arrays + colors | Vertices: (N,3), Faces: (M,3), Colors: (N,3) |
| **Mesh File**   | M8 (Exporter)          | M10 (Visualization)    | .obj ASCII            | Wavefront format                             |
| **Timing Data** | M9 (Profiler)          | M10 (Analysis)         | JSON/CSV              | {"stage": time_ms, ...}                      |

---

### 4.2.8 Verification and Simulation Plan

This section defines the functional verification strategy, specifying test protocols, acceptance criteria, and evidence collection methods for each design alternative.

#### Verification Objectives

1. **Correctness Validation**: Ensure each design produces geometrically valid, visually plausible meshes
2. **Performance Verification**: Confirm speedup claims and throughput targets are met
3. **Regression Testing**: Detect unintended changes when modifying code
4. **Reproducibility**: Verify outputs are deterministic across multiple runs

#### Design A Verification Plan

**VA1: End-to-End Pipeline Test**

- **Objective**: Verify Design A processes AFLW2000 images and generates meshes
- **Inputs**: 5 representative images (varied poses)
- **Execution**: Run `./scripts/test_single_aflw2000.sh` per image
- **Acceptance Criteria**:
  - All 5 images produce .obj files
  - Mesh files load in MeshLab without errors
  - Vertex counts are in expected range (30K–50K post-merge)
  - Processing time is 11–18 seconds per image
- **Evidence**: Screenshots of meshes in MeshLab, processing logs

**VA2: Batch Processing Test**

- **Objective**: Verify batch processing of 50-image subset
- **Execution**: Run `./scripts/batch_process_aflw2000.sh`
- **Acceptance Criteria**:
  - 43 successful meshes generated
  - 7 face detection failures logged correctly
  - No crashes or container errors
  - batch_process.log contains timing for all 50 images
- **Evidence**: Log files, file count verification

**VA3: Mesh Validity Test**

- **Objective**: Validate geometric properties of output meshes
- **Test Cases**:
  - Load mesh in trimesh: `mesh = trimesh.load('output.obj')`
  - Check `mesh.is_watertight` (should be True for well-formed meshes)
  - Verify coordinate ranges (X: 40–140, Y: 30–180, Z: 15–90)
  - Check RGB colors present (not all black/white)
- **Acceptance Criteria**:
  - At least 90% of meshes are watertight
  - Coordinate ranges match expected face dimensions
  - Colors are plausible (skin tones visible)
- **Evidence**: Automated validation script output

**VA4: Failure Mode Test**

- **Objective**: Verify pipeline handles failures gracefully
- **Test Cases**:
  - Process image with no face (profile view >90°)
  - Process corrupted image file
  - Process image with multiple faces
- **Acceptance Criteria**:
  - Pipeline logs error message (does not crash)
  - Processing time <2 seconds for failed cases
  - No partial output files created
- **Evidence**: Error logs, terminal output

#### Design B Verification Plan

**VB1: CUDA Extension Build Test**

- **Objective**: Verify CUDA extension compiles and loads
- **Execution**:
  ```bash
  cd designB/cuda_kernels
  python setup.py build_ext --inplace
  python -c "import marching_cubes_cuda_ext; print('OK')"
  ```
- **Acceptance Criteria**:
  - Compilation succeeds without errors
  - Shared library (.so file) created
  - Python import succeeds
- **Evidence**: Build logs, import success message

**VB2: Synthetic Volume Test (Unit Test)**

- **Objective**: Verify CUDA kernel correctness on known geometry
- **Test Case**: Generate synthetic sphere volume (128³), run marching cubes
- **Execution**:
  ```python
  volume = generate_sphere(radius=30, resolution=128)
  verts_cpu, faces_cpu = marching_cubes_cpu(volume, threshold=0.5)
  verts_gpu, faces_gpu = marching_cubes_gpu(volume, threshold=0.5)
  assert np.allclose(verts_cpu, verts_gpu, atol=1e-3)
  ```
- **Acceptance Criteria**:
  - GPU and CPU vertex counts match within 1%
  - GPU speedup >5× observed
  - Sphere mesh is visually correct (smooth, closed surface)
- **Evidence**: Test script output, sphere mesh screenshot

**VB3: VRN Volume Regression Test**

- **Objective**: Verify GPU marching cubes matches CPU baseline on real volumes
- **Test Cases**: Process 10 VRN volumes with both CPU and GPU
- **Metrics**:
  - Vertex count difference (should be <5% after post-processing)
  - Coordinate range overlap (should be >95%)
  - Visual similarity (manual inspection)
- **Acceptance Criteria**:
  - All 10 volumes produce non-empty meshes
  - Post-processed vertex counts match within tolerance
  - No obvious visual artifacts (inverted faces, holes)
- **Evidence**: Comparison table, side-by-side mesh screenshots

**VB4: Performance Benchmark Test**

- **Objective**: Verify speedup and throughput claims
- **Execution**: Run `designB/scripts/run_benchmarks.py` on 43 volumes
- **Measurements**:
  - CPU time per volume (3 runs, average)
  - GPU time per volume (3 runs, average)
  - Speedup = CPU_time / GPU_time
  - Throughput = 1 / GPU_time (volumes/sec)
- **Acceptance Criteria**:
  - Average speedup ≥16× (measured: 18.36×)
  - Throughput ≥50 volumes/sec (measured: 217 vol/sec)
  - Standard deviation <10% of mean
  - GPU memory <100 MB per volume
- **Evidence**: Benchmark report (JSON), timing plots

**VB5: Stability and Reliability Test**

- **Objective**: Verify GPU pipeline handles edge cases and repeated execution
- **Test Cases**:
  - Process same volume 100 times (check determinism)
  - Process empty volume (all zeros)
  - Process volume with extreme threshold (T=0.1, T=0.9)
  - Run on multi-GPU system (specify device_id=1)
- **Acceptance Criteria**:
  - Outputs are bit-exact across 100 runs
  - Empty volume produces empty mesh (no crash)
  - Extreme thresholds log warnings but do not crash
  - Device selection works correctly
- **Evidence**: Determinism test logs, edge case output files

**VB6: Post-Processing Correctness Test**

- **Objective**: Verify axis transforms and color mapping are correct
- **Test Cases**:
  - Load raw marching cubes mesh, apply transforms manually
  - Compare with automated post-processing output
  - Verify Z-axis range is reduced by 50%
  - Check RGB colors match input image regions
- **Acceptance Criteria**:
  - Manual and automated transforms produce identical outputs
  - Z-scaling factor verified (range before/after ≈ 2:1)
  - Colors map to correct face regions (eyes, nose, mouth)
- **Evidence**: Coordinate range statistics, color distribution histograms

#### Design C Verification Plan (Preliminary)

**VC1: FaceScape Data Loader Test**

- **Objective**: Verify FaceScape images and GT meshes load correctly
- **Test Cases**:
  - Load 5 samples from FaceScape subset
  - Check image dimensions, GT mesh vertex counts
  - Verify metadata (subject ID, expression) parsed correctly
- **Acceptance Criteria**:
  - All 5 samples load without errors
  - Image dimensions match expected resolution
  - GT meshes are watertight (trimesh validation)
- **Evidence**: Data loader test script output

**VC2: Error Metric Validation Test**

- **Objective**: Verify Chamfer distance and Hausdorff distance implementations
- **Test Cases**:
  - Compute metrics on synthetic meshes (sphere, translated sphere)
  - Verify distance equals known values (e.g., sphere translation distance)
- **Acceptance Criteria**:
  - Chamfer distance matches analytical solution within 1%
  - Hausdorff distance matches analytical solution within 1%
- **Evidence**: Metric validation test output

**VC3: End-to-End FaceScape Test**

- **Objective**: Verify Design C processes FaceScape samples and computes metrics
- **Execution**: Process 5 FaceScape samples, compute error metrics vs. GT
- **Acceptance Criteria**:
  - All 5 samples produce valid meshes
  - Error metrics are in plausible range (Chamfer <10mm)
  - Error heatmaps generated successfully
- **Evidence**: Output meshes, error metric JSON, heatmap images

#### Acceptance Criteria Summary

| Test ID | Design | Criterion            | Target                                       |
| ------- | ------ | -------------------- | -------------------------------------------- |
| VA1     | A      | End-to-end success   | 5/5 samples generate meshes                  |
| VA2     | A      | Batch success rate   | 43/50 (86%)                                  |
| VA3     | A      | Mesh validity        | ≥90% watertight                              |
| VB1     | B      | CUDA build           | Compilation succeeds, import OK              |
| VB2     | B      | Synthetic test       | GPU vs CPU vertex match <1%                  |
| VB3     | B      | Regression test      | 10/10 volumes match CPU baseline             |
| VB4     | B      | Performance          | Speedup ≥16×, throughput ≥50 vol/sec         |
| VB5     | B      | Stability            | 100 runs bit-exact, edge cases handled       |
| VB6     | B      | Post-processing      | Z-scaling correct, colors aligned            |
| VC1     | C      | FaceScape loader     | 5/5 samples load correctly                   |
| VC2     | C      | Metric validation    | Chamfer/Hausdorff within 1% of analytical    |
| VC3     | C      | End-to-end FaceScape | 5/5 samples produce meshes, metrics computed |

#### Evidence Collection

**Logs and Reports**:

- Build logs: `designB/cuda_kernels/build/*.log`
- Processing logs: `data/out/designA/batch_process.log`, `data/out/designB/processing.log`
- Benchmark reports: `designB/benchmarks/benchmark_results.json`
- Timing data: `data/out/designA/time.log`, `designB/benchmarks/timing_*.csv`

**Visual Artifacts**:

- Mesh screenshots: `results/verification/meshes/*.png`
- Comparison plots: `results/verification/plots/*.png` (CPU vs GPU timing, speedup distribution)
- Error heatmaps (Design C): `results/verification/facescape_errors/*.png`

**Automated Test Scripts**:

- `tests/test_design_a.py` - Design A validation suite
- `tests/test_design_b.py` - Design B validation suite (unit + integration)
- `tests/test_design_c.py` - Design C validation suite (preliminary)
- `tests/test_modules.py` - Module-level unit tests (M1–M10)

---

### 4.2.9 Implementation Environment and Reproducibility

This section specifies the hardware, software, and environmental dependencies required to reproduce the VRN pipeline implementations.

#### Hardware Requirements

**Minimum Configuration**:

- **CPU**: x86_64, 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB (dataset + Docker images + outputs)
- **GPU** (Design B only): NVIDIA GPU with compute capability ≥5.0 (Maxwell or newer)

**Tested Configuration**:

- **CPU**: AMD Ryzen 7 5800X (8 cores, 16 threads, 3.8 GHz)
- **RAM**: 32 GB DDR4-3200
- **GPU**: NVIDIA GeForce RTX 4070 SUPER (12 GB GDDR6X, SM 8.9 Ada Lovelace)
- **Storage**: 1 TB NVMe SSD

#### Software Environment

**Operating System**:

- **Design A**: Ubuntu 22.04 LTS (host), Ubuntu 16.04 (Docker container)
- **Design B**: Ubuntu 22.04 LTS
- **Kernel**: 5.15.0 or newer

**Docker** (Design A):

- **Version**: 24.0.5
- **Image**: `asjackson/vrn:latest` (publicly available)
- **Image Size**: ~4 GB

**Python Environment** (Design B):

- **Python**: 3.10.12
- **Package Manager**: pip 23.0
- **Virtual Environment**: venv (Python standard library)

**Python Dependencies** (from `requirements.txt`):

```
torch==2.1.0+cu118
numpy==1.26.4
scikit-image==0.22.0
trimesh==4.0.5
Pillow==10.1.0
scipy==1.11.4
matplotlib==3.8.2
```

**CUDA Toolchain** (Design B):

- **CUDA Toolkit**: 11.8 (tested), 12.x (compatible)
- **Compute Capability Target**: SM 8.6 (Ampere, compiled target)
- **Runtime Compatibility**: SM 8.9 (Ada Lovelace) via forward binary compatibility
- **Driver**: NVIDIA Driver 470+ (supports CUDA 11.8)
- **cuDNN**: Not required (custom kernel does not use cuDNN)

**Build Tools** (Design B CUDA Extension):

- **C++ Compiler**: GCC 9.4.0 (Ubuntu default)
- **CUDA Compiler**: nvcc 11.8 (from CUDA Toolkit)
- **Build System**: setuptools 68.0, ninja 1.11 (optional, accelerates build)
- **C++ Standard**: C++17 (required by PyTorch 2.x)

#### Compilation Flags (Design B)

**NVCC Flags** (specified in `setup.py`):

```python
extra_compile_args = {
    'cxx': ['-std=c++17', '-O3'],
    'nvcc': [
        '-std=c++17',
        '-O3',
        '--use_fast_math',
        '-arch=sm_86',  # Target Ampere (max for CUDA 11.8)
        '-gencode=arch=compute_86,code=sm_86'
    ]
}
```

**Explanation**:

- `-std=c++17`: Required by PyTorch 2.x headers (uses `if constexpr`, `std::is_same_v`)
- `-O3`: Maximum optimization
- `--use_fast_math`: Enable fast approximations for math functions (slight precision trade-off)
- `-arch=sm_86`: Generate code for SM 8.6 (Ampere, RTX 3000 series)
- Forward compatibility: SM 8.6 code runs on SM 8.9 (Ada Lovelace, RTX 4000 series) without recompilation

#### Directory Structure and File Organization

```
VRN/
├── data/
│   ├── in/
│   │   ├── aflw2000/              # AFLW2000 subset (50 images)
│   │   └── facescape/             # FaceScape dataset (Design C, planned)
│   ├── out/
│   │   ├── designA/               # Design A outputs (meshes, logs)
│   │   ├── designB/               # Design B outputs
│   │   │   ├── meshes/            # Final .obj files
│   │   │   ├── volumes/           # Exported .npy volumes
│   │   │   └── logs/              # Processing and benchmark logs
│   │   └── designC/               # Design C outputs (planned)
│   └── tmp/                       # Temporary files (can be deleted)
├── scripts/
│   ├── batch_process_aflw2000.sh  # Design A batch script
│   ├── test_single_aflw2000.sh    # Design A single-image test
│   ├── analyze_results.sh         # Metrics generation
│   └── check_status.sh            # Progress monitoring
├── designB/
│   ├── cuda_kernels/              # CUDA extension source
│   │   ├── marching_cubes_kernel.cu
│   │   ├── marching_cubes_bindings.cpp
│   │   ├── marching_cubes_tables.h
│   │   ├── cuda_marching_cubes.py
│   │   └── setup.py
│   ├── python/
│   │   ├── convert_raw_to_npy.py  # Volume format conversion
│   │   ├── marching_cubes_cpu.py  # CPU baseline implementation
│   │   └── mesh_postprocess.py    # Transform, merge, color mapping
│   ├── scripts/
│   │   ├── build_cuda_extension.sh
│   │   ├── run_pipeline.sh
│   │   └── run_benchmarks.py
│   └── benchmarks/
│       ├── benchmark_results.json
│       └── timing_plots/
├── docs/
│   ├── Chapter_4.1_Design_Methodology.md
│   ├── Chapter_4.2_Preliminary_Design_Specification.md  # This document
│   ├── designA_metrics.md
│   ├── DesignB_Benchmark_Results.md
│   ├── aflw2000_subset.txt
│   └── ...
├── tests/
│   ├── test_design_a.py
│   ├── test_design_b.py
│   ├── test_design_c.py
│   └── test_modules.py
├── results/
│   ├── verification/
│   │   ├── meshes/
│   │   ├── plots/
│   │   └── logs/
│   └── poster/
│       └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

#### Reproducibility Instructions

**Design A Reproduction**:

1. **Environment Setup**:

   ```bash
   # Install Docker
   sudo apt install docker.io
   sudo usermod -aG docker $USER
   # Log out and back in

   # Pull VRN image
   docker pull asjackson/vrn:latest
   ```

2. **Dataset Preparation**:

   ```bash
   # Copy AFLW2000 images to data/in/aflw2000/
   # Use file list in docs/aflw2000_subset.txt
   ```

3. **Execution**:

   ```bash
   # Single image test
   ./scripts/test_single_aflw2000.sh

   # Full batch (43 meshes expected)
   ./scripts/batch_process_aflw2000.sh

   # Generate metrics
   ./scripts/analyze_results.sh
   ```

4. **Verification**:

   ```bash
   # Check outputs
   ls data/out/designA/*.obj | wc -l  # Should output: 43

   # Load mesh in MeshLab
   meshlab data/out/designA/image00002.jpg.obj
   ```

**Design B Reproduction**:

1. **Environment Setup**:

   ```bash
   # Create virtual environment
   python3 -m venv vrn_env
   source vrn_env/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Verify CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Build CUDA Extension**:

   ```bash
   cd designB/cuda_kernels
   python setup.py build_ext --inplace
   python -c "import marching_cubes_cuda_ext; print('OK')"
   ```

3. **Volume Preparation**:

   ```bash
   # Export volumes from Design A (if not already done)
   # Or use pre-exported .npy files in data/out/volumes/
   ```

4. **Execution**:

   ```bash
   # Process single volume
   python designB/python/marching_cubes_cuda.py \
     --input data/out/volumes/image00002.npy \
     --output data/out/designB/meshes/image00002.obj

   # Run full pipeline
   ./designB/scripts/run_pipeline.sh

   # Run benchmarks
   python designB/scripts/run_benchmarks.py
   ```

5. **Verification**:

   ```bash
   # Check outputs
   ls data/out/designB/meshes/*.obj | wc -l  # Should output: 43

   # Verify benchmark results
   cat designB/benchmarks/benchmark_results.json
   ```

#### Determinism and Seeding

**Random Seed Configuration**:

- **Face Detection**: dlib is deterministic (no randomness)
- **VRN Inference**: Torch7 forward pass is deterministic
- **Marching Cubes**: Deterministic algorithm (no randomness)
- **Vertex Merging**: Octree-based merging is deterministic given fixed tolerance

**Non-Deterministic Sources** (none expected):

- GPU floating-point operations may have minor rounding differences (<1e-6) across runs
- Parallel atomic operations in CUDA kernel are ordered but timing-dependent (should not affect output)

**Verification**: Run same input 10 times, compare outputs bit-by-bit:

```bash
for i in {1..10}; do
  python designB/python/marching_cubes_cuda.py --input test.npy --output out_$i.obj
done
md5sum out_*.obj  # All checksums should match
```

---

### 4.2.10 Summary

This chapter has provided a comprehensive engineering specification for the VRN 3D face reconstruction pipeline, defining three design alternatives (A, B, C) with detailed module decomposition, interface specifications, and verification protocols.

#### Key Specifications Established

1. **System Architecture**: Five-stage pipeline (face detection → VRN inference → volume export → marching cubes → post-processing) with well-defined stage boundaries and data interfaces.

2. **Design A (Legacy CPU Baseline)**: Fully specified Docker-based pipeline with measured performance baseline (10.26s/image, 86% success rate on AFLW2000 subset).

3. **Design B (Hybrid CPU+CUDA)**: Two-stage architecture with custom CUDA marching cubes kernel (8×8×8 thread blocks, SM 8.6 target) achieving 18.36× speedup over CPU baseline while producing 69% higher-resolution meshes (63,571 vs. 37,587 vertices post-merge).

4. **Design C (FaceScape Generalization)**: Preliminary specification for dataset adaptation with quantitative evaluation (Chamfer distance, Hausdorff distance) against ground-truth 3D scans.

5. **Module Decomposition**: Ten well-defined modules (M1–M10) with clear responsibilities, dependencies, and failure modes.

6. **Interfaces**: Standardized data formats (volumes: 200×192×192 float32 .npy, meshes: Wavefront .obj with RGB colors) and configuration parameters.

7. **Verification Plan**: Comprehensive test protocols (VA1–VA4 for Design A, VB1–VB6 for Design B, VC1–VC3 for Design C) with quantitative acceptance criteria and evidence collection methods.

8. **Implementation Environment**: Complete toolchain specification (Ubuntu 22.04, Python 3.10, PyTorch 2.1.0, CUDA 11.8, GCC 9.4) with reproducibility instructions.

#### Traceability to Chapter 4.1

All functional (FR1–FR5), performance (PR1–PR5), and quality (QR1–QR5) requirements from Section 4.1.2 are satisfied by at least one design alternative, as documented in Section 4.2.2. Verification methods for each requirement are specified in Section 4.2.8.

#### Foundation for Results and Poster

This specification provides the technical foundation for:

- **Chapter 5 (Results)**: Performance measurements, mesh quality comparisons, and ablation studies will reference the benchmarking protocols defined in Section 4.2.8.

- **Poster Presentation**: Visual comparisons (Design A vs. B meshes), timing plots, and error heatmaps (Design C) will use outputs generated according to the specifications in this chapter.

- **Future Work**: Design C's preliminary specification (Section 4.2.5) establishes a clear roadmap for extending the pipeline to FaceScape with quantitative evaluation.

The engineering rigor applied in this specification ensures that all subsequent results are reproducible, traceable to requirements, and grounded in validated implementations.

---

**End of Chapter 4.2**
