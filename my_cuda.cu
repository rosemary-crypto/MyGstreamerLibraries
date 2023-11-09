#include "my_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelFunction() {
    // CUDA kernel code
}

// Implement cu_function
void cu_function() {
    std::cout << "cu_function from my_cuda.cu" << std::endl;

    // Call the CUDA kernel
    kernelFunction << <1, 1 >> > ();
    cudaDeviceSynchronize(); // Ensure CUDA operations complete
}