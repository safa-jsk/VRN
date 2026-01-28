"""
Setup script for CUDA Marching Cubes Extension
Compiles CUDA kernel and creates Python module
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Cannot build CUDA extension.")

print(f"Building for CUDA {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# Get compute capability
device_capability = torch.cuda.get_device_capability()
# RTX 4070 SUPER is SM 8.9 (Ada), but CUDA 11.8 only supports up to SM 8.6
# Use SM 8.6 for backward compatibility (Ampere max)
sm_version = "sm_86"  # Max supported by CUDA 11.8
print(f"Target GPU compute capability: sm_{device_capability[0]}{device_capability[1]} (using {sm_version} for CUDA 11.8)")

# CUDA kernel sources
cuda_sources = [
    'cuda_kernels/marching_cubes_kernel.cu',
    'cuda_kernels/marching_cubes_bindings.cpp'
]

# Compiler flags
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],  # C++17 required for PyTorch 2.x
    'nvcc': [
        '-O3',
        '--use_fast_math',
        f'-arch={sm_version}',
        '--expt-relaxed-constexpr',
        '-std=c++17',  # C++17 for CUDA
        '-Xcompiler', '-fPIC'
    ]
}

setup(
    name='marching_cubes_cuda',
    version='1.0.0',
    description='Custom CUDA Marching Cubes for PyTorch',
    author='VRN Research',
    ext_modules=[
        CUDAExtension(
            name='marching_cubes_cuda_ext',
            sources=cuda_sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[os.path.abspath('cuda_kernels')]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        f'torch>={torch.__version__}',
        'numpy>=1.24.0'
    ],
    python_requires='>=3.7'
)
