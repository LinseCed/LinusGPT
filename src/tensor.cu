#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void fill_kernel(float* data, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

Tensor::Tensor(size_t rows, size_t cols) : rows(rows), cols(cols), d_data(nullptr) {
    data = new float[rows * cols];

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    use_gpu = (err == cudaSuccess && deviceCount > 0);

    if (use_gpu) {
        cudaMalloc(&d_data, rows * cols * sizeof(float));
        std::cout << "Using GPU\n";
    } else {
        std::cout << "Using CPU\n";
    }
}

Tensor::~Tensor() {
    delete[] data;
    if (use_gpu) cudaFree(d_data);
}

void Tensor::fill_cpu(float value) {
    for (size_t i = 0; i < rows * cols; i++) data[i] = value;
}

void Tensor::fill(float value) {
    if (use_gpu) {
        size_t size = rows * cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
    }
}
