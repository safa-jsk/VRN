import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import torch

# Use PyTorch's CUDA toolkit instead of system nvcc
torch_cuda_home = os.path.dirname(torch.__file__)

setup(
    name='chamfer',
    ext_modules=[
        CUDAExtension(
            'chamfer', 
            [
                'chamfer_cuda.cpp',
                'chamfer.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': [
                    '-O3',
                    '-std=c++14',
                    # Use compute_80 for compatibility with CUDA 11.5 nvcc
                    # RTX 40 series will use PTX JIT compilation
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_80,code=compute_80',  # PTX for forward compat
                    '-Xcompiler', '-fPIC',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    })
