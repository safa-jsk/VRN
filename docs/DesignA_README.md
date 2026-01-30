# Design A - Implementation Guide

This directory contains scripts and data for the Design A baseline implementation of VRN (CPU-only Docker pipeline).

## Quick Start

### Single Image Test

Test with one AFLW2000 image:

```bash
./scripts/test_single_aflw2000.sh
```

### Full Batch Processing

Process all 50 AFLW2000 images:

```bash
./scripts/batch_process_aflw2000.sh
```

**Note:** This will take approximately 15-20 minutes (50 images × ~18 seconds each).

### Analyze Results

Generate metrics report:

```bash
./scripts/analyze_results.sh
```

## Directory Structure

```
VRN/
├── data/
│   ├── in/
│   │   ├── turing.jpg              # Demo image
│   │   └── aflw2000/               # AFLW2000-3D test subset (50 images)
│   ├── out/
│   │   ├── turing.jpg.obj          # Demo mesh output
│   │   ├── turing.jpg.crop.jpg     # Demo crop output
│   │   └── aflw2000/               # Batch outputs (meshes, crops, logs)
│   └── tmp/                        # Scratch space
├── scripts/
│   ├── test_single_aflw2000.sh     # Quick test (1 image)
│   ├── batch_process_aflw2000.sh   # Full batch processing
│   └── analyze_results.sh          # Generate metrics report
├── results/
│   └── poster/
│       └── meshes/                 # Curated mesh screenshots for poster
└── docs/
    └── designA_metrics.md          # Baseline metrics report
```

## Expected Outputs

For each input image `imageXXXXX.jpg`, VRN generates:

- `imageXXXXX.jpg.obj` - 3D mesh in Wavefront OBJ format (~4-5 MB)
- `imageXXXXX.jpg.crop.jpg` - Aligned and cropped face (450×450 px)

## Performance Baseline

- **Processing time:** ~18-20 seconds per image (CPU-only)
- **Success rate:** Depends on face detection (frontal/near-frontal faces work best)
- **Output size:** ~4-5 MB per mesh, ~37.6K vertices after post-processing

## Viewing Results

### Using MeshLab

```bash
# Install MeshLab if needed
sudo apt install meshlab

# Open a mesh
meshlab data/out/aflw2000/image00002.jpg.obj
```

### Using Python (alternative)

```python
import trimesh
mesh = trimesh.load('data/out/aflw2000/image00002.jpg.obj')
mesh.show()
```

## Troubleshooting

### Docker permission error

If you get "permission denied" errors:

```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

### No face detected

Some images may fail if:

- Face is not frontal (profile views)
- Face is too small or occluded
- Extreme lighting conditions

Check the logs in `data/out/aflw2000/batch_process.log` for details.

## Next Steps for Thesis

1. **Run full batch** - Process all 50 images
2. **Select examples** - Choose 6-10 representative meshes for poster
3. **Render in MeshLab** - Create consistent screenshots (front/3-4/side views)
4. **Document observations** - Note success/failure patterns
5. **Write Chapter 4** - Methodology and preliminary design sections

## Batch Processing Options

### Option 1: Process all 50 images now

```bash
time ./scripts/batch_process_aflw2000.sh
```

Estimated time: ~15-20 minutes

### Option 2: Process in smaller batches

```bash
# Process first 10 only
for f in $(ls data/in/aflw2000/*.jpg | head -10); do
    docker run --rm -v "$PWD/data:/data" asjackson/vrn:latest \
        /runner/run.sh "/data/in/aflw2000/$(basename "$f")"
done
```

### Option 3: Overnight processing for larger subset

If you want to process more images (100-200) for better statistics:

```bash
# Copy more images
find data/AFLW2000 -name "*.jpg" | sort | head -200 | \
    xargs -I {} cp {} data/in/aflw2000/

# Run batch (will take ~1 hour for 200 images)
nohup ./scripts/batch_process_aflw2000.sh > batch_run.log 2>&1 &
```

## Design A Success Criteria

- [x] Environment set up (Docker image pulled)
- [x] Single image demo working (turing.jpg → .obj)
- [x] Batch processing pipeline created
- [x] Test run completed (1/50 images)
- [ ] Full batch completed
- [ ] Metrics documented
- [ ] Poster screenshots prepared

---

**Status:** Ready for full batch processing. Run `./scripts/batch_process_aflw2000.sh` when ready.
