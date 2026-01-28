#!/bin/bash
# Run Full Design B Benchmarks and Comparison
# Benchmarks CUDA marching cubes vs CPU baseline on all AFLW2000 volumes

set -e

cd /home/ahad/Documents/VRN/designB
source ../vrn_env/bin/activate

echo "=========================================="
echo "Design B - Full Benchmark Suite"
echo "=========================================="
echo ""

# Step 1: Run benchmarks
echo "STEP 1: Running CPU vs GPU Benchmarks"
echo "--------------------------------------"
echo "Processing 43 volumes with 3 runs each..."
echo ""

python3 python/benchmarks.py \
    --volumes ../data/out/designB/volumes \
    --output ../data/out/designB/benchmarks_cuda \
    --runs 3 \
    --plot

echo ""
echo ""

# Step 2: Generate comparison report
echo "STEP 2: Generating Design Comparison Report"
echo "--------------------------------------------"
python3 python/compare_designs.py

echo ""
echo ""

# Step 3: Display results
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""

# Show comparison if it exists
if [ -f "docs/Design_Comparison.md" ]; then
    cat docs/Design_Comparison.md
else
    echo "⚠ Comparison report not generated"
fi

echo ""
echo "=========================================="
echo "Files Generated"
echo "=========================================="
echo ""
echo "Benchmarks:"
echo "  - data/out/designB/benchmarks_cuda/benchmark_results.json"
echo "  - data/out/designB/benchmarks_cuda/benchmark_plots.png"
echo ""
echo "Reports:"
echo "  - docs/Design_Comparison.md"
echo "  - results/comparisons/design_comparison.png"
echo ""
echo "✓ Benchmark suite complete!"
