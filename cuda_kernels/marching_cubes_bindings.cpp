/*
 * PyTorch C++ Bindings for CUDA Marching Cubes
 * Bridges CUDA kernel to Python via pybind11
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of CUDA kernel launcher
extern "C" void launchMarchingCubes(
    const float* d_volume,
    float* d_vertices,
    int* d_triangles,
    int* d_numVertices,
    int* d_numTriangles,
    float isolevel,
    int dimX, int dimY, int dimZ,
    int maxVertices, int maxTriangles,
    cudaStream_t stream
);

// PyTorch wrapper
torch::Tensor marching_cubes_forward(
    torch::Tensor volume,
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor num_vertices,
    torch::Tensor num_triangles,
    float isolevel,
    int dimX, int dimY, int dimZ,
    int maxVertices, int maxTriangles
) {
    // Verify inputs are on CUDA
    TORCH_CHECK(volume.is_cuda(), "volume must be a CUDA tensor");
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(triangles.is_cuda(), "triangles must be a CUDA tensor");
    
    // Use default CUDA stream (0) for simplicity
    cudaStream_t stream = 0;
    
    // Launch kernel
    launchMarchingCubes(
        volume.data_ptr<float>(),
        vertices.data_ptr<float>(),
        triangles.data_ptr<int>(),
        num_vertices.data_ptr<int>(),
        num_triangles.data_ptr<int>(),
        isolevel,
        dimX, dimY, dimZ,
        maxVertices, maxTriangles,
        stream
    );
    
    // Synchronize
    cudaStreamSynchronize(stream);
    
    return vertices;  // Return for chaining
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes_forward", &marching_cubes_forward, 
          "CUDA Marching Cubes Forward Pass");
}
