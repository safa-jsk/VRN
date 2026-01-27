#!/bin/bash
# Quick status check for Design A processing

echo "======================================"
echo "Design A - Processing Status"
echo "======================================"
echo ""

# Input status
total_input=$(ls -1 data/in/aflw2000/*.jpg 2>/dev/null | wc -l)
echo "📥 Input Images: ${total_input}"
echo ""

# Output status
meshes=$(ls -1 data/out/designA/*.obj 2>/dev/null | wc -l)
crops=$(ls -1 data/out/designA/*.crop.jpg 2>/dev/null | wc -l)
echo "📤 Generated Outputs:"
echo "   - Meshes (.obj):     ${meshes}"
echo "   - Crops (.crop.jpg): ${crops}"
echo ""

# Progress
if [ ${total_input} -gt 0 ]; then
    progress=$(awk "BEGIN {printf \"%.1f\", (${meshes}/${total_input})*100}")
    echo "📊 Progress: ${meshes}/${total_input} (${progress}%)"
else
    echo "📊 Progress: 0/0 (0%)"
fi
echo ""

# Recent activity
if [ -f "data/out/designA/batch_process.log" ]; then
    echo "📝 Recent Activity (last 5 lines):"
    tail -5 data/out/designA/batch_process.log
    echo ""
fi

# File sizes
if [ ${meshes} -gt 0 ]; then
    total_size=$(du -sh data/out/designA/*.obj 2>/dev/null | tail -1 | awk '{print $1}')
    echo "💾 Total mesh output size: ${total_size}"
    echo ""
fi

# Next steps
if [ ${meshes} -lt ${total_input} ] && [ ${total_input} -gt 0 ]; then
    remaining=$((total_input - meshes))
    estimated_time=$((remaining * 19))  # ~19 seconds per image
    estimated_min=$((estimated_time / 60))
    echo "⏱️  Estimated time remaining: ~${estimated_min} minutes"
    echo ""
    echo "▶️  To continue processing, run:"
    echo "   ./scripts/batch_process_aflw2000.sh"
elif [ ${meshes} -eq ${total_input} ] && [ ${total_input} -gt 0 ]; then
    echo "✅ Batch processing complete!"
    echo ""
    echo "▶️  Next steps:"
    echo "   1. Run: ./scripts/analyze_results.sh"
    echo "   2. Review: docs/designA_metrics.md"
    echo "   3. Select meshes for poster visualization"
else
    echo "⚠️  No input images found in data/in/aflw2000/"
fi

echo ""
echo "======================================"
