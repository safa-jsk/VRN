#!/bin/bash
# Clean up incorrect Design B data
# Removes synthetic volumes that were created from Design A meshes

echo "Cleaning up incorrect Design B implementation..."
echo ""

# Backup old data (optional)
if [ -d "data/out/designB/volumes" ]; then
    echo "Backing up old (incorrect) volumes..."
    mv data/out/designB/volumes data/out/designB/volumes_old_synthetic
    echo "  → Moved to: data/out/designB/volumes_old_synthetic/"
fi

if [ -d "data/out/designB/meshes" ]; then
    echo "Backing up old meshes..."
    mv data/out/designB/meshes data/out/designB/meshes_old
    echo "  → Moved to: data/out/designB/meshes_old/"
fi

# Create fresh directories
mkdir -p data/out/designB/volumes
mkdir -p data/out/designB/volumes_raw
mkdir -p data/out/designB/meshes

echo ""
echo "✓ Cleanup complete"
echo ""
echo "Ready to run correct Design B pipeline:"
echo "  ./scripts/designB_run_correct.sh"
