# Design A - Baseline Metrics

**Date:** 2026-01-28 00:41:00  
**Docker Image:** asjackson/vrn:latest  
**Hardware:** CPU-only (no GPU acceleration)

---

## Processing Summary

| Metric | Value |
|--------|-------|
| Total Input Images | 50 |
| Successfully Processed | 43 |
| Failed | 7 |
| Success Rate | 86.00% |

---

## Output Files Generated

- **Meshes (*.obj):** 43
- **Cropped faces (*.crop.jpg):** 43

---

## Timing Statistics (per image)

| Statistic | Time (seconds) |
|-----------|----------------|
| Average | 10.26 |
| Minimum | 1 |
| Maximum | 18 |

**Note:** Timing includes Docker container startup overhead, face detection, volumetric regression, and isosurface extraction. Failed images (no face detected) complete in 1-2 seconds, while successful reconstructions take 11-18 seconds.

---

## Observations

### Successful Processing
- VRN successfully generates 3D face meshes from 2D images
- Output format: Wavefront OBJ files with vertex and face data
- Each mesh contains ~440K vertices

### Limitations Identified
- CPU-only processing is relatively slow (11-18 seconds per successful image)
- Face detection failures: 7 out of 50 images (14%) failed to detect faces
- Common failure modes:
  - Extreme side poses (profile views beyond ~±90°)
  - Heavy occlusion
  - Very low resolution images
  - Images where dlib face detector cannot locate face region
  
### Pipeline Stages
1. **Face Detection:** dlib-based detector locates face region
2. **Alignment & Crop:** Standardizes input to 450×450 pixels
3. **Volume Regression:** CNN predicts 3D volumetric representation
4. **Isosurface Extraction:** Marching cubes algorithm generates mesh

---

## File Locations

- **Input images:** `data/in/aflw2000/`
- **Output meshes:** `data/out/designA/*.obj`
- **Cropped faces:** `data/out/designA/*.crop.jpg`
- **Processing log:** `data/out/designA/batch_process.log`
- **Timing data:** `data/out/designA/time.log`

---

## Next Steps for Poster

1. Select 6-10 representative examples (varied poses, lighting)
2. Load meshes in MeshLab for visualization
3. Capture screenshots from multiple angles (front, 3/4 view, side)
4. Create comparison grid: input photo → 3D reconstruction

---

## Design A Deliverables Status

- [x] Folder structure established
- [x] Single-image demo verified (turing.jpg)
- [x] Batch processing script created
- [x] AFLW2000-3D subset processed (43 meshes generated)
- [x] Baseline metrics documented
- [x] Batch processing completed with 86.00% success rate
- [ ] Poster-ready mesh screenshots (to be generated in MeshLab)
- [ ] Chapter 4 methodology section (in progress)

---

## Failed Images Analysis

The following images failed to produce mesh outputs:

- image00004.jpg
- image00010.jpg
- image00021.jpg
- image00032.jpg
- image00036.jpg
- image00074.jpg
- image00075.jpg

**Failure Pattern:** All failures occurred at the face detection stage (processing completed in 1-2 seconds), indicating the dlib detector could not locate a face in these images. This is typically due to extreme poses, occlusion, or unusual image characteristics.

