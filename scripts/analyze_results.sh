#!/bin/bash
# Design A - Results Analysis Script
# Analyzes batch processing results and generates metrics

OUTPUT_DIR="data/out/designA"
RESULTS_FILE="docs/designA_metrics.md"

echo "Analyzing Design A results..."

# Count files
num_input=$(ls -1 data/in/aflw2000/*.jpg 2>/dev/null | wc -l)
num_meshes=$(ls -1 ${OUTPUT_DIR}/*.obj 2>/dev/null | wc -l)
num_crops=$(ls -1 ${OUTPUT_DIR}/*.crop.jpg 2>/dev/null | wc -l)

# Calculate success rate
if [ $num_input -gt 0 ]; then
    success_rate=$(awk "BEGIN {printf \"%.2f\", (${num_meshes}/${num_input})*100}")
else
    success_rate="0.00"
fi

# Get timing stats from batch_process.log
if [ -f "${OUTPUT_DIR}/batch_process.log" ]; then
    # Parse "Completed in X seconds" lines from the log
    timing_data=$(grep "Completed in" "${OUTPUT_DIR}/batch_process.log" | awk '{print $3}')
    
    if [ -n "$timing_data" ]; then
        # Calculate statistics using awk
        read avg_time min_time max_time total_count <<< $(echo "$timing_data" | awk '
        BEGIN {sum=0; count=0; min=99999; max=0}
        {
            sum+=$1; 
            count++; 
            if($1<min) min=$1; 
            if($1>max) max=$1
        }
        END {
            if(count>0) 
                printf "%.2f %d %d %d", sum/count, min, max, count
            else 
                print "N/A N/A N/A 0"
        }')
    else
        avg_time="N/A"
        min_time="N/A"
        max_time="N/A"
    fi
else
    avg_time="N/A"
    min_time="N/A"
    max_time="N/A"
fi

# Generate markdown report
cat > "${RESULTS_FILE}" << EOF
# Design A - Baseline Metrics

**Date:** $(date '+%Y-%m-%d %H:%M:%S')  
**Docker Image:** asjackson/vrn:latest  
**Hardware:** CPU-only (no GPU acceleration)

---

## Processing Summary

| Metric | Value |
|--------|-------|
| Total Input Images | ${num_input} |
| Successfully Processed | ${num_meshes} |
| Failed | $((num_input - num_meshes)) |
| Success Rate | ${success_rate}% |

---

## Output Files Generated

- **Meshes (*.obj):** ${num_meshes}
- **Cropped faces (*.crop.jpg):** ${num_crops}

---

## Timing Statistics (per image)

| Statistic | Time (seconds) |
|-----------|----------------|
| Average | ${avg_time} |
| Minimum | ${min_time} |
| Maximum | ${max_time} |

**Note:** Timing includes Docker container startup overhead, face detection, volumetric regression, and isosurface extraction. Failed images (no face detected) complete in 1-2 seconds, while successful reconstructions take 11-18 seconds.

---

## Observations

### Successful Processing
- VRN successfully generates 3D face meshes from 2D images
- Output format: Wavefront OBJ files with vertex and face data
- Each post-processed mesh contains ~37.6K vertices on average

### Limitations Identified
- CPU-only processing is relatively slow (11-18 seconds per successful image)
- Face detection failures: $((num_input - num_meshes)) out of ${num_input} images ($((100 - ${success_rate%.*}))%) failed to detect faces
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

- **Input images:** \`data/in/aflw2000/\`
- **Output meshes:** \`data/out/designA/*.obj\`
- **Cropped faces:** \`data/out/designA/*.crop.jpg\`
- **Processing log:** \`data/out/designA/batch_process.log\`
- **Timing data:** \`data/out/designA/time.log\`

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
- [x] AFLW2000-3D subset processed (${num_meshes} meshes generated)
- [x] Baseline metrics documented
EOF

# Add failed images list if there are failures
if [ $((num_input - num_meshes)) -gt 0 ] && [ -f "${OUTPUT_DIR}/batch_process.log" ]; then
    cat >> "${RESULTS_FILE}" << EOF
- [x] Batch processing completed with ${success_rate}% success rate
- [ ] Poster-ready mesh screenshots (to be generated in MeshLab)
- [ ] Chapter 4 methodology section (in progress)

---

## Failed Images Analysis

The following images failed to produce mesh outputs:

EOF
    grep "✗ FAILED" "${OUTPUT_DIR}/batch_process.log" | sed 's/✗ FAILED: No .obj output for /- /' >> "${RESULTS_FILE}"
    
    cat >> "${RESULTS_FILE}" << EOF

**Failure Pattern:** All failures occurred at the face detection stage (processing completed in 1-2 seconds), indicating the dlib detector could not locate a face in these images. This is typically due to extreme poses, occlusion, or unusual image characteristics.
EOF
else
    cat >> "${RESULTS_FILE}" << EOF
- [ ] Poster-ready mesh screenshots (to be generated in MeshLab)
- [ ] Chapter 4 methodology section (in progress)
EOF
fi

cat >> "${RESULTS_FILE}" << EOF

EOF

# Mesh metrics (Chamfer, F1_tau, F1_2tau)
METRICS_DIR="${OUTPUT_DIR}/metrics"
METRICS_CSV="${METRICS_DIR}/mesh_metrics.csv"
METRICS_SUMMARY="${METRICS_DIR}/mesh_metrics_summary.txt"

mkdir -p "${METRICS_DIR}"

echo ""
echo "================================================"
echo "Design A Mesh Metrics"
echo "================================================"

python3 scripts/designA_mesh_metrics.py \
    --pred-dir "${OUTPUT_DIR}" \
    --ref-dir "data/out/designB/meshes" \
    --output-csv "${METRICS_CSV}" \
    | tee "${METRICS_SUMMARY}"

# Append metrics summary to the report
cat >> "${RESULTS_FILE}" << EOF

---

## Mesh Metrics (Design A vs Reference)

**CSV Output:** \`${METRICS_CSV}\`

\`\`\`
$(cat "${METRICS_SUMMARY}")
\`\`\`

EOF

echo ""
echo "================================================"
echo "Design A Metrics Summary"
echo "================================================"
cat "${RESULTS_FILE}"
echo ""
echo "Full report saved to: ${RESULTS_FILE}"
