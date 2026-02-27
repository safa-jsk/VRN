# Chapter 4: Design and Implementation

## 4.1 Design Process and Methodology Overview

This chapter presents the engineering design methodology employed to develop and evaluate alternative implementations of the VRN (Volumetric Regression Network) 3D face reconstruction pipeline. The primary objective of this work is to accelerate mesh extraction through GPU parallelization while preserving reconstruction quality and maintaining compatibility with the legacy VRN inference system. This section establishes the systematic design process, requirements analysis, alternative generation strategy, experimental validation protocol, and comparative evaluation framework that guided the implementation and assessment of multiple pipeline variants.

### 4.1.1 Design Objectives and Motivation

The overarching goal of this project is to improve the computational efficiency of the VRN pipeline for 3D face reconstruction while maintaining or enhancing mesh quality. The original VRN implementation, based on legacy Torch7 framework and CPU-only processing, provides high-quality volumetric predictions but suffers from significant latency in the mesh extraction stage. As 3D face reconstruction increasingly targets real-time applications—including interactive avatar generation, augmented reality facial tracking, and rapid dataset annotation—the need for acceleration becomes critical.

#### Primary Objectives

The design objectives are formulated as follows:

1. **GPU Acceleration**: Leverage CUDA parallel computing capabilities to accelerate the computationally intensive marching cubes algorithm, which extracts isosurfaces from volumetric predictions. This represents the primary performance bottleneck in the post-processing pipeline.

2. **Throughput Improvement**: Achieve measurable reduction in per-volume processing time to enable batch processing of large-scale datasets (hundreds to thousands of faces) within practical time constraints. Target throughput should approach real-time capability (≥30 volumes per second).

3. **Latency Reduction**: Minimize end-to-end pipeline latency to support interactive applications where user feedback is time-critical. While VRN inference remains CPU-bound in the legacy container, reducing mesh extraction latency contributes to overall responsiveness.

4. **Quality Preservation**: Ensure that GPU-accelerated implementations produce geometrically equivalent or superior meshes compared to the CPU baseline. Quality metrics include mesh topology (vertex/face counts), geometric fidelity (coordinate accuracy), and visual plausibility (smooth surfaces, correct color mapping).

5. **Reproducibility and Determinism**: Maintain deterministic behavior across multiple runs to enable fair performance comparisons and consistent experimental evaluation. All implementations must produce identical outputs given identical inputs.

6. **Modularity and Maintainability**: Design pipeline components with clear interfaces to facilitate independent testing, validation, and future extensions (e.g., alternative mesh extraction algorithms or dataset adaptations).

#### Practical Motivation

Performance improvements in 3D mesh reconstruction have direct practical implications:

- **Research Iteration Speed**: Faster processing enables rapid experimentation with algorithm variants, hyperparameters, and dataset subsets, accelerating the research development cycle.
- **Production Deployment**: Real-time or near-real-time reconstruction is essential for interactive applications such as virtual try-on systems, telepresence avatars, and gaming character creation.

- **Dataset Preprocessing**: Large-scale 3D face datasets (e.g., VGGFace2, CelebA, FaceScape) require efficient batch processing to generate training data for downstream tasks such as face recognition, expression analysis, and 3D morphable model fitting.

- **Hardware Efficiency**: Effective GPU utilization reduces computational costs and energy consumption compared to underutilized CPU-only processing, which is particularly relevant for cloud-based or edge deployment scenarios.

---

### 4.1.2 System Requirements and Constraints

The design process is constrained by functional, performance, and quality requirements, as well as technical and environmental limitations inherited from the legacy VRN system.

#### Functional Requirements

The pipeline must satisfy the following functional specifications:

- **FR1: End-to-End Mesh Generation**: Accept 2D face images as input and produce 3D mesh files (.obj format) as output, including vertex coordinates, face connectivity, and RGB color attributes.

- **FR2: Dataset Compatibility**: Process AFLW2000-3D images (and potentially other datasets such as FaceScape) with consistent preprocessing (face detection, alignment, cropping).

- **FR3: Format Interoperability**: Generate meshes compatible with standard 3D visualization tools (MeshLab, Blender) and downstream analysis pipelines (PyTorch3D, Open3D).

- **FR4: Batch Processing Support**: Enable automated batch execution over multiple images with logging, error handling, and progress monitoring.

- **FR5: Hardware Portability**: Support execution on available hardware platforms (Ubuntu 22.04, NVIDIA RTX 4070 SUPER with CUDA 11.8/12.8 drivers).

#### Performance Requirements

Quantitative performance targets guide the design of accelerated implementations:

- **PR1: Speedup Objective**: Achieve minimum 10× speedup on the marching cubes stage compared to CPU baseline (scikit-image implementation).

- **PR2: Throughput Target**: Process at least 50 volumes/second on target GPU hardware to enable real-time capability.

- **PR3: Latency Consistency**: Maintain low variance in processing time across different volumes (standard deviation <10% of mean).

- **PR4: Memory Efficiency**: Limit GPU memory allocation to <100 MB per volume to enable concurrent batch processing.

- **PR5: Benchmarking Rigor**: Measure performance using statistically valid protocols (multiple runs, warm-up iterations, mean/std reporting).

#### Quality Requirements

Output quality is assessed through the following criteria:

- **QR1: Geometric Correctness**: Meshes must exhibit correct spatial scaling (X, Y, Z axis ranges consistent with facial proportions), proper coordinate system orientation, and accurate vertex positions relative to input face features.

- **QR2: Topological Consistency**: Mesh connectivity must represent a valid manifold surface with no self-intersections, degenerate triangles, or disconnected components.

- **QR3: Visual Plausibility**: Rendered meshes should display smooth surfaces without visible artifacts (staircasing, holes, excessive faceting) when viewed in standard mesh visualization tools.

- **QR4: Color Fidelity**: RGB color mapping from input images to mesh vertices must preserve facial appearance, including skin tone, texture boundaries, and lighting variations.

- **QR5: Quantitative Proxy Metrics**: Since ground-truth 3D scans are not available for all test images, mesh resolution (vertex count) and extraction consistency (threshold behavior) serve as proxy indicators of quality.

#### System Constraints

The design space is constrained by the following technical and environmental limitations:

- **C1: Legacy VRN Stack**: The VRN neural network inference is implemented in Torch7 (released 2016) with CUDA 7.5 and cuDNN 5.1 dependencies. These versions are incompatible with modern NVIDIA GPUs (Turing, Ampere, Ada Lovelace architectures). Consequently, VRN inference must remain CPU-bound within the legacy Docker container (asjackson/vrn:latest).

- **C2: Limited Retraining Feasibility**: Retraining or fine-tuning the VRN model is out of scope due to unavailability of original training data, computational cost, and project timeline constraints. The focus is on accelerating post-processing rather than modifying the network.

- **C3: Dataset Format Heterogeneity**: AFLW2000-3D uses MATLAB .mat files for annotations, while FaceScape uses JSON metadata. Preprocessing pipelines must accommodate different annotation schemas.

- **C4: GPU Compute and Memory Limits**: Target hardware (RTX 4070 SUPER, 12GB VRAM, SM 8.9) imposes memory bandwidth and compute throughput ceilings. CUDA kernel implementations must respect these physical constraints.

- **C5: Compilation Toolchain Versioning**: CUDA 11.8 supports compute capabilities up to SM 8.6 (Ampere), requiring forward compatibility for Ada Lovelace (SM 8.9) execution. PyTorch 2.1.0 requires C++17 standard for extension compilation.

- **C6: Reproducibility and Determinism**: All processing stages must produce identical outputs across multiple runs to enable fair performance comparisons. Non-deterministic behaviors (e.g., random sampling, race conditions) must be eliminated.

---

### 4.1.3 Design Process Methodology

The engineering design process follows a structured workflow adapted from systems engineering best practices:

**Phase 1: Requirements Analysis** → **Phase 2: Alternative Design Generation** → **Phase 3: Implementation and Verification** → **Phase 4: Experimental Validation** → **Phase 5: Comparative Evaluation** → **Phase 6: Design Selection**

This iterative methodology enables systematic exploration of the design space while maintaining clear traceability from requirements to final implementation.

#### Phase 1: Requirements Analysis and Profiling

The design process begins with profiling the baseline VRN pipeline to identify performance bottlenecks. Using Python's `cProfile` and manual timing instrumentation, the pipeline is decomposed into constituent stages:

1. Face detection (dlib): ~0.1–0.5 s
2. Image preprocessing and alignment: ~0.05 s
3. VRN volumetric regression (Torch7, CPU): ~2–3 s
4. Volume export and format conversion: ~0.05 s
5. Marching cubes mesh extraction (CPU): ~0.085 s
6. Mesh post-processing (trimming, merging, color mapping): ~0.1–0.2 s

**Bottleneck Identification**: VRN inference (stage 3) dominates total latency (>80%). However, due to constraint C1 (legacy stack incompatibility), this stage cannot be accelerated without significant architectural redesign. The marching cubes stage (stage 5), while contributing only ~3–4% to total latency, is computationally intensive, highly parallelizable, and amenable to GPU acceleration. This analysis motivates the selection of marching cubes as the primary optimization target.

#### Phase 2: Modular Pipeline Design Rationale

Given the constraints, a **two-stage modular architecture** is adopted:

- **Stage 1: VRN Inference (CPU-bound, legacy)**: Processes input images through the Docker container to generate 200×192×192 volumetric predictions. Outputs are exported as .raw (binary) or .npy (NumPy) files.

- **Stage 2: Mesh Extraction (GPU-accelerated, modern)**: Reads pre-computed volumes and applies CUDA-accelerated marching cubes to generate meshes. This stage is implemented using PyTorch 2.1.0 + CUDA 11.8/12.8 with custom C++ extensions.

**Design Justification**:

- **Separation of Concerns**: Isolates legacy code (Torch7) from modern tooling (PyTorch + CUDA 11.8+), minimizing compatibility conflicts.
- **Incremental Acceleration**: Enables performance gains in stage 2 without requiring full pipeline rewrite.

- **Reusability**: Pre-computed volumes can be reprocessed with different thresholds, algorithms, or hyperparameters without re-running expensive VRN inference.

- **Experimental Flexibility**: Facilitates direct comparison between CPU and GPU implementations of marching cubes using identical input data.

- **Future Extensibility**: Provides a clean interface for replacing stage 1 with a modernized VRN implementation (e.g., PyTorch port) in future work.

---

### 4.1.4 Alternative Design Generation and Rationale

Three alternative pipeline designs are defined to explore the trade-off space between performance, quality, implementation complexity, and dataset generalization.

#### Design A: CPU-Only Baseline (Legacy Pipeline)

**Description**: The original VRN Docker pipeline running entirely on CPU, processing images end-to-end without GPU acceleration. Uses scikit-image's marching_cubes function for isosurface extraction.

**Architecture**:

```
Input Image → Docker Container (Torch7 + dlib) → VRN Inference (CPU) →
Marching Cubes (scikit-image, CPU) → Output Mesh (.obj)
```

**Purpose and Rationale**:

- **Baseline Reference**: Establishes ground-truth correctness for mesh geometry, topology, and visual appearance. All accelerated designs must produce geometrically equivalent or superior outputs.

- **Performance Baseline**: Provides reference latency and throughput metrics for quantifying speedup achieved by GPU acceleration.

- **Failure Mode Analysis**: Identifies inherent limitations of the pipeline (e.g., face detection failures) independent of acceleration strategy.

- **Reproducibility**: Uses publicly available Docker image (asjackson/vrn:latest), ensuring reproducible evaluation environment.

**Dataset**: AFLW2000-3D subset (50 images, manually selected, documented in `docs/aflw2000_subset.txt`).

**Expected Outcomes**: Successful mesh generation for frontal and near-frontal faces; failures on extreme poses (±90° yaw), severe occlusions, and poor lighting conditions.

#### Design B: CUDA-Accelerated Marching Cubes (GPU Pipeline)

**Description**: Replaces the CPU-based marching cubes implementation with a custom CUDA kernel, integrated into PyTorch via C++ extensions (pybind11). VRN inference remains in the legacy Docker container, but volumes are exported and processed by the GPU pipeline.

**Architecture**:

```
Stage 1: Input Image → Docker Container → VRN Inference (CPU) → Volume (.npy)
Stage 2: Volume (.npy) → Custom CUDA Marching Cubes (GPU) → Output Mesh (.obj)
```

**Technical Implementation**:

- **CUDA Kernel**: Implements parallel marching cubes algorithm with thread-per-voxel parallelism. Thread block configuration: 8×8×8 threads (512 threads/block).
- **Compilation Target**: SM 8.6 (maximum supported by CUDA 11.8 toolchain), with forward compatibility to SM 8.9 (Ada Lovelace architecture).

- **Integration**: PyTorch C++ extension using `CUDAExtension` API, enabling seamless tensor interoperability between Python and CUDA.

- **Volume Format**: 200×192×192 float32 tensors, with boolean thresholding applied (threshold = 0.5).

- **Mesh Post-Processing**: Axis permutation (Z,Y,X → X,Y,Z), Z-axis scaling (×0.5), vertex merging (tolerance = 0.1), and RGB color mapping from input volumes.

**Design Rationale**:

- **High Parallelism**: Marching cubes is embarrassingly parallel (each voxel processed independently), making it ideal for GPU acceleration.

- **Custom Control**: Implementing a custom kernel (rather than using third-party libraries like PyTorch3D or NVIDIA Kaolin) provides fine-grained control over memory access patterns, lookup table optimization, and output formatting.

- **Learning Objective**: Demonstrates CUDA proficiency and low-level GPU programming skills, which are core competencies for this thesis.

- **Compatibility**: PyTorch integration ensures smooth interoperability with modern Python scientific computing stack.

**Expected Outcomes**: Minimum 10× speedup over CPU baseline; mesh quality equivalent or superior (due to reduced merging tolerance); throughput sufficient for real-time processing.

#### Design C: Dataset Generalization (FaceScape Domain Shift)

**Description**: Extends Design B to process FaceScape dataset, which contains high-quality 3D scans with diverse expressions and identities. This design evaluates the generalization capability of the VRN model and GPU pipeline across different data distributions.

**Planned Modifications**:

- **Preprocessing Adaptation**: FaceScape images require different cropping/alignment strategies (JSON-based landmark annotations rather than MATLAB .mat files).

- **Evaluation Protocol**: Comparison against FaceScape ground-truth 3D scans using geometric error metrics (Chamfer distance, Hausdorff distance, normal consistency).

- **Quality Assessment**: Quantitative validation using reference geometry, rather than qualitative proxy metrics (vertex counts).

**Rationale**:

- **Domain Shift Analysis**: AFLW2000-3D consists primarily of in-the-wild images with varied quality, while FaceScape provides controlled studio captures. Evaluating performance across both datasets assesses robustness.

- **Future Work Direction**: Establishes a pathway for integrating modern datasets and evaluation protocols into the accelerated pipeline.

**Status**: Planned but not executed in current scope (deferred to future work section).

#### Summary Table: Design Alternatives

| Design       | Pipeline Architecture             | Marching Cubes Implementation | Dataset                  | Primary Objective                                |
| ------------ | --------------------------------- | ----------------------------- | ------------------------ | ------------------------------------------------ |
| **Design A** | Docker (CPU-only)                 | scikit-image (CPU)            | AFLW2000-3D (50 images)  | Baseline correctness + performance reference     |
| **Design B** | Docker (Stage 1) + CUDA (Stage 2) | Custom CUDA kernel (GPU)      | AFLW2000-3D (43 volumes) | GPU acceleration + quality improvement           |
| **Design C** | Same as Design B                  | Same as Design B              | FaceScape                | Dataset generalization + quantitative validation |

---

### 4.1.5 Experimental Design and Functional Verification

The experimental methodology follows a controlled evaluation protocol designed to ensure reproducibility, statistical validity, and fair comparison across design alternatives.

#### Evaluation Dataset: AFLW2000-3D Subset

**Dataset Specification**:

- **Source**: Annotated Facial Landmarks in the Wild 3D (AFLW2000-3D), a subset of AFLW containing 2,000 images with 68 3D facial landmarks.
- **Subset Selection**: First 50 images (alphabetically ordered filenames: `image00002.jpg` through `image00075.jpg`), documented in `docs/aflw2000_subset.txt`.

- **Characteristics**: Diverse head poses (yaw: -90° to +90°, pitch: -30° to +30°), varied lighting conditions, indoor/outdoor scenes, mixed demographics.

**Rationale for Subset Size**: The 50-image subset balances statistical representativeness with practical processing time for iterative development. This size provides sufficient diversity to identify failure modes while enabling rapid experimentation.

#### Success and Failure Criteria

**Design A (End-to-End Pipeline)**:

- **Success**: Mesh file (.obj) generated with valid geometry (>1000 vertices, no NaN coordinates).
- **Failure**: No mesh output due to face detection failure (dlib cannot locate face bounding box).

**Design B (Volume-to-Mesh Pipeline)**:

- **Success**: Mesh generated from pre-computed volume with consistent topology and color attributes.
- **Failure**: Invalid volume (all zeros, corrupted format), or mesh extraction errors (empty output, degenerate geometry).

#### Verification Stages

Functional verification proceeds through three stages:

**Stage 1: Sanity Checks**

- **File Existence**: Verify that output files (.obj meshes, .crop.jpg images) are created.
- **Format Validity**: Load meshes in standard tools (MeshLab, Trimesh) without parsing errors.

- **Non-Degeneracy**: Confirm vertex count >1000 and face count >100 (indicating meaningful geometry).

**Stage 2: Correctness Checks**

During Design B implementation, six critical bugs were identified and fixed through systematic correctness validation:

1. **Volume Type Bug**: Initial implementation used float32 volumes instead of boolean thresholding, extracting only 0.8% of expected voxels. Fixed by applying `volume.astype(bool)` before marching cubes.

2. **Threshold Value Bug**: Incorrect threshold (10.0 on boolean volume) caused 10× over-extraction. Corrected to threshold = 0.5.

3. **Missing RGB Colors**: Initial meshes lacked color attributes. Fixed by adding RGB mapping from input volume coordinates to image pixels.

4. **Transformation Order Bug**: Colors were mapped before spatial transformation, causing vertex reordering inconsistencies. Fixed by transforming vertices first, then mapping colors.

5. **Z-Axis Compression**: Output meshes had narrow Z-range (41.5 units instead of 71). Fixed by applying Z-scaling factor (×0.5) after axis permutation.

6. **Aggressive Vertex Merging**: Merge tolerance of 1.0 reduced vertex count to ~31,000 (too coarse). Reduced tolerance to 0.1, preserving ~63,000 vertices.

These bugs were discovered through comparative analysis with Design A outputs (visual inspection in MeshLab, bounding box comparison, histogram analysis of coordinate distributions).

**Stage 3: Performance Benchmarking**

Timing measurements follow this protocol:

- **Warm-up**: 5 iterations to load CUDA kernels and initialize GPU context.
- **Measurement Runs**: 3 independent executions per volume, averaging results.

- **Metrics Collected**: Execution time (seconds), standard deviation, speedup factor (relative to CPU baseline).

- **Environment Control**: Single-user mode, background processes minimized, fixed GPU clock rates (no dynamic frequency scaling).

#### Verification as "Simulation"

In the context of this thesis, **"simulation"** refers to controlled experimental evaluation of alternative designs for functional verification. Unlike hardware simulation (e.g., RTL testbenches), this refers to:

- **Synthetic Test Cases**: Processing known-good volumes with predictable characteristics (e.g., unit sphere) to validate kernel correctness.

- **Regression Testing**: Re-running Design A baseline on the same dataset to confirm output consistency across software versions.

- **Ablation Studies**: Isolating individual pipeline stages (e.g., marching cubes only, without post-processing) to measure component-level performance.

This methodology ensures that observed performance improvements are attributable to design changes rather than confounding factors (different datasets, hardware variations, measurement errors).

---

### 4.1.6 Benchmarking Protocol

Quantitative performance evaluation follows a rigorous benchmarking protocol to ensure reproducible and statistically valid measurements.

#### Hardware and Software Environment

**Hardware**:

- **GPU**: NVIDIA GeForce RTX 4070 SUPER (12GB GDDR6X VRAM, Ada Lovelace architecture, SM 8.9 compute capability)
- **CPU**: AMD Ryzen 7 5800X (8 cores, 16 threads, 3.8 GHz base clock)

- **Memory**: 32GB DDR4-3200 RAM

**Software**:

- **Operating System**: Ubuntu 22.04 LTS (kernel 5.15.0)

- **CUDA Toolkit**: 11.8 (compilation target: SM 8.6) and 12.8 (runtime driver)

- **PyTorch**: 2.1.0 with CUDA 11.8 backend

- **Python**: 3.10.12

- **Docker**: 24.0.5 (for Design A VRN container)

#### Timing Instrumentation

**CPU Baseline (Design A)**:

```python
import time
t_start = time.perf_counter()
vertices, faces = marching_cubes(volume, threshold=0.5)
t_end = time.perf_counter()
cpu_time = t_end - t_start
```

**GPU Implementation (Design B)**:

```python
import torch
torch.cuda.synchronize()  # Wait for previous operations
t_start = time.perf_counter()
vertices, faces = marching_cubes_cuda(volume_tensor, threshold=0.5)
torch.cuda.synchronize()  # Ensure kernel completion
t_end = time.perf_counter()
gpu_time = t_end - t_start
```

**Critical Detail**: `torch.cuda.synchronize()` calls are essential to prevent asynchronous kernel execution from distorting timing measurements.

#### Measurement Protocol

1. **Warm-up Phase**: Execute 5 dummy iterations to initialize CUDA context, load kernels, and stabilize GPU clock frequencies.

2. **Measurement Phase**: For each volume:
   - Load volume from disk into memory (not timed).
   - Execute marching cubes (CPU or GPU).
   - Record execution time.
   - Repeat 3 times per volume.
   - Compute mean and standard deviation.

3. **Aggregation**: Average results across all volumes (n=43 for Design B, n=50 for Design A).

4. **Speedup Calculation**:
   ```
   Speedup = CPU_time_mean / GPU_time_mean
   ```

#### Controlled Variables

To ensure fair comparison:

- **Identical Volumes**: Design B processes the same 43 volumes that succeeded in Design A.
- **Same Threshold**: Both CPU and GPU implementations use threshold = 0.5.

- **Consistent Post-Processing**: Axis permutation, Z-scaling, vertex merging, and RGB mapping applied identically to both outputs.

- **Single-Threaded CPU Baseline**: scikit-image marching_cubes runs on single core to isolate algorithm performance (multi-threading would confound comparison).

---

### 4.1.7 Comparative Results Summary (Methodology-Driven Evidence)

This subsection presents high-level comparative metrics derived from the experimental methodology. Detailed numerical results and visualizations are deferred to subsequent results chapters (if applicable).

#### Design A: Baseline Performance and Success Rate

**Processing Summary**:

- **Total Input Images**: 50 (AFLW2000-3D subset)
- **Successful Mesh Generations**: 43 (86.0% success rate)

- **Failures**: 7 images (14.0%) due to face detection failures (dlib unable to locate face region)

**Timing Characteristics** (per image, end-to-end):

- **Average**: 10.26 seconds
- **Minimum**: 1 second (detection failure, early exit)

- **Maximum**: 18 seconds (successful reconstruction)

**Failure Analysis**: Failed images exhibited extreme head poses (near-profile views, ±90° yaw), heavy occlusions (hands, scarves), or low resolution (<100×100 pixels after cropping).

#### Design B: GPU Acceleration Performance

**Marching Cubes Latency** (per volume, 200×192×192 voxels):

- **CPU Baseline** (scikit-image): 84.2 ms average (σ = 3.1 ms)

- **GPU Implementation** (custom CUDA): 4.6 ms average (σ = 0.2 ms)

- **Speedup**: **18.36× mean** (range: 16.51× to 19.58×)

**Throughput**:

- **CPU**: ~11.9 volumes/second

- **GPU**: ~217 volumes/second

**GPU Resource Utilization**:

- **VRAM Allocated**: ~28 MB per volume

- **Peak Memory Reserved**: 1,720 MB (includes PyTorch overhead)

- **Utilization Efficiency**: Excellent (low overhead, high parallelism)

#### Mesh Quality Comparison

**Mesh Resolution** (average vertex counts, post-processed after merge):

| Design       | Average Vertices | Average Faces | Improvement         |
| ------------ | ---------------- | ------------- | ------------------- |
| **Design A** | 37,587           | 125,290       | Baseline            |
| **Design B** | 63,571           | 211,903       | **+69.0%** vertices |

**Note**: Raw marching-cubes outputs (pre-merge) are higher; the CUDA benchmark reports an average of 172,317 vertices (≈145k–220k range) before post-processing.

**Interpretation**: Design B produces significantly higher-resolution post-processed meshes due to reduced vertex merging tolerance (0.1 vs 1.0). This results in smoother surfaces and better preservation of fine facial details (e.g., nose bridge curvature, lip contours).

**Geometric Correctness**: Visual inspection in MeshLab confirms that Design B meshes exhibit correct spatial scaling (X: 45-133, Y: 33-175, Z: 17-88 typical ranges), proper RGB color mapping (skin tones, hair colors preserved), and no topological defects (manifold surfaces, no self-intersections).

#### Pipeline Acceleration Impact

**Marching Cubes Stage** (Design B):

- **Absolute Time Saved**: 79.6 ms per volume (84.2 ms → 4.6 ms)

- **Percentage of Total Pipeline Time** (Design A): ~3.3% (marching cubes) out of ~2.5 seconds (VRN inference dominates)

**End-to-End Impact**:

- **Design A Total**: ~10.26 seconds per image (averaged over successes and failures)

- **Design B Total** (hypothetical, if volumes are pre-computed): ~0.4 seconds per volume (marching cubes + post-processing)

**Critical Observation**: Because VRN inference remains CPU-bound (constraint C1), the overall end-to-end speedup is modest (~3-4%). However, for workflows where volumes are pre-computed or cached (e.g., hyperparameter tuning, algorithm ablation studies), Design B provides substantial acceleration.

#### Summary Table: Key Performance Metrics

| Metric                              | Design A (CPU) | Design B (GPU)            | Improvement     |
| ----------------------------------- | -------------- | ------------------------- | --------------- |
| Success Rate                        | 43/50 (86%)    | 43/43 (100%)<sup>\*</sup> | N/A<sup>†</sup> |
| Avg. Processing Time (full)         | 10.26 s/image  | N/A<sup>‡</sup>           | —               |
| Marching Cubes Latency              | 84.2 ms        | 4.6 ms                    | **18.36×**      |
| Throughput (MC only)                | 11.9 vol/s     | 217 vol/s                 | **18.2×**       |
| GPU Memory Usage                    | 0 MB           | 28 MB                     | +28 MB          |
| Avg. Mesh Vertices (post-processed) | 37,587         | 63,571                    | **+69.0%**      |
| Avg. Mesh Faces (post-processed)    | 125,290        | 211,903                   | **+69.1%**      |
| Real-Time Capable?                  | ❌ (12 FPS)    | ✅ (217 FPS)              | Yes             |

**Note**: Vertex and face counts in this table refer to post-processed meshes after axis transform and vertex merging.

<sup>\*</sup> Design B processes only the 43 successful volumes from Design A (failures excluded at preprocessing stage).

<sup>†</sup> Success rate comparison not directly applicable; Design B starts from validated volumes.

<sup>‡</sup> Design B end-to-end timing not measured due to Docker overhead in Stage 1.

---

### 4.1.8 Threats to Validity and Limitations

The experimental methodology and design choices introduce several limitations that must be acknowledged to ensure proper interpretation of results.

#### Internal Validity

**Threat 1: VRN Inference Not Accelerated**

- **Description**: The VRN volumetric regression stage remains CPU-bound in the legacy Docker container, dominating total pipeline latency (~80% of end-to-end time).

- **Impact**: Overall pipeline speedup is limited to ~3-4% despite 18× acceleration of marching cubes. End-to-end performance gains are modest unless volumes are pre-computed.

- **Mitigation**: Results are reported with clear separation of stage-level performance (marching cubes only) vs. end-to-end performance (full pipeline). Claims of speedup are scoped appropriately.

**Threat 2: Dataset Subset Size**

- **Description**: Evaluation uses 50-image subset of AFLW2000-3D rather than full 2,000-image dataset.

- **Impact**: Statistical representativeness may be limited; outlier cases or rare failure modes might not be captured.

- **Mitigation**: Subset includes diverse poses, lighting, and demographics. Failure analysis (7/50 = 14%) aligns with expected dlib face detection limitations. Findings are generalized cautiously.

**Threat 3: Timing Measurement Precision**

- **Description**: Sub-millisecond timing variations due to OS scheduling, GPU frequency scaling, and background processes.

- **Impact**: Standard deviation of GPU timings (±0.2 ms) represents 4.3% of mean (4.6 ms).

- **Mitigation**: Multiple measurement runs (n=3 per volume), warm-up iterations, and controlled environment (single-user mode) reduce noise. Statistical aggregation provides robust estimates.

#### External Validity

**Threat 4: Design B Processes Volumes, Not End-to-End Images**

- **Description**: Design B evaluation assumes pre-computed volumes, bypassing face detection and VRN inference stages.

- **Impact**: Direct comparison of success rates (86% for Design A vs. 100% for Design B) is misleading; Design B operates only on validated volumes.

- **Mitigation**: Results clearly state that Design B processes 43 successful volumes from Design A. Failure cases are attributed to Stage 1 (face detection) rather than Stage 2 (mesh extraction).

**Threat 5: Generalization to Other Datasets**

- **Description**: Evaluation limited to AFLW2000-3D; performance on FaceScape, VGGFace2, or other datasets is untested.

- **Impact**: Observed speedups and quality improvements may not generalize to different image distributions (e.g., higher resolution, controlled lighting, synthetic data).

- **Mitigation**: Design C (planned future work) explicitly addresses dataset generalization by incorporating FaceScape. Current findings are scoped to AFLW2000-3D.

#### Construct Validity

**Threat 6: Quality Metrics are Proxy Measures**

- **Description**: Mesh quality assessed using vertex counts and visual inspection rather than ground-truth geometric error (Chamfer distance, normal consistency).

- **Impact**: Higher vertex counts correlate with smoother surfaces but do not guarantee geometric accuracy relative to true 3D face shape.

- **Mitigation**: Design A outputs serve as correctness reference (validated by original VRN authors). Design B improvements (higher resolution) are confirmed through visual inspection in MeshLab. Quantitative error metrics deferred to Design C (FaceScape evaluation).

**Threat 7: Single Hardware Configuration**

- **Description**: Benchmarks conducted on single GPU model (RTX 4070 SUPER); performance on other architectures (Turing, Ampere, professional GPUs) is unknown.

- **Impact**: Absolute speedup values (18.36×) may vary with GPU generation, memory bandwidth, and SM count.

- **Mitigation**: Relative speedup trends (GPU >> CPU for embarrassingly parallel workloads) are expected to generalize across CUDA-capable hardware. Absolute numbers are reported with hardware specification.

---

### 4.1.9 Conclusion: Methodology Outcome and Design Selection

The systematic design methodology presented in this chapter provides a validated foundation for evaluating and selecting among alternative VRN pipeline implementations. Through structured requirements analysis, modular architecture design, rigorous experimental protocols, and comparative benchmarking, the following conclusions are established:

1. **Design A (CPU Baseline)** successfully processes 43 of 50 AFLW2000-3D images (86% success rate), providing a reproducible correctness reference for mesh geometry, topology, and visual quality. Failure cases are attributable to inherent limitations of face detection (extreme poses, occlusions) rather than pipeline deficiencies.

2. **Design B (GPU-Accelerated Marching Cubes)** achieves a **18.36× average speedup** over CPU baseline on the mesh extraction stage, demonstrating effective CUDA parallelization. Throughput increases from 12 volumes/second (CPU) to 217 volumes/second (GPU), enabling real-time processing. Additionally, Design B produces **69% higher-resolution post-processed meshes** (63,571 vs. 37,587 vertices) due to refined post-processing parameters.

3. **Functional Verification** confirmed that Design B meshes exhibit correct geometric scaling, valid topology, accurate RGB color mapping, and visual quality equivalent or superior to Design A. Six critical implementation bugs were identified and resolved through systematic correctness validation.

4. **Limitations Acknowledged**: VRN inference remains CPU-bound (constraint C1), limiting overall end-to-end speedup to ~3-4%. However, for workflows involving pre-computed volumes—such as algorithm development, hyperparameter optimization, and batch reprocessing—Design B provides substantial acceleration.

5. **Design Selection**: Design B is adopted as the improved implementation for Part 2 of this thesis, based on measurable performance gains (18× speedup), quality improvements (+69% vertices), and successful validation against Design A baseline.

6. **Future Work Motivation**: The methodology establishes a pathway for Design C (dataset generalization with FaceScape), which will incorporate quantitative geometric error metrics and evaluate robustness across different data distributions.

The modular pipeline architecture, combined with rigorous experimental protocols, ensures that observed improvements are reproducible, attributable to specific design decisions, and generalizable within the scope of stated constraints. This methodology demonstrates engineering rigor in design evaluation and provides a template for future extensions of the VRN pipeline.

---

**End of Chapter 4.1**
