#!/bin/bash
# Build CUDA Marching Cubes Extension
# For VRN Design B GPU Acceleration

set -e

echo "=========================================="
echo "Building CUDA Marching Cubes Extension"
echo "=========================================="

# Activate virtual environment
if [ -d "vrn_env" ]; then
    echo "✓ Activating vrn_env..."
    source vrn_env/bin/activate
else
    echo "✗ Error: vrn_env not found"
    echo "  Please run from VRN root directory"
    exit 1
fi

# Check Python environment
echo ""
echo "Python Environment:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# Check CUDA compiler
echo ""
echo "CUDA Compiler:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "  ✓ nvcc found"
else
    echo "  ✗ nvcc not found"
    echo "    Install CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    exit 1
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf cuda_kernels/__pycache__
echo "  ✓ Cleaned"

# Build extension
echo ""
echo "Compiling CUDA kernel..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ CUDA Extension Built Successfully"
    echo "=========================================="
    
    # Test the extension
    echo ""
    echo "Testing extension..."
    python3 -c "from cuda_kernels.cuda_marching_cubes import marching_cubes_gpu; print('✓ Import successful')"
    
    echo ""
    echo "Ready to run Design B pipeline with GPU acceleration!"
    echo "Execute: ./scripts/designB_run.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ Build Failed"
    echo "=========================================="
    echo "Check error messages above"
    exit 1
fi
