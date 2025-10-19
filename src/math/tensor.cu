#include "math/tensor.h"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
bool use_gpu = false;

__global__ void fill_kernel(float* data, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matadd_kernel(const float* A, const float* B, float* C, size_t rows, size_t cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

Tensor::Tensor() : Tensor(0, 0) {}

Tensor::Tensor(size_t rows, size_t cols) : rows(rows), cols(cols), d_data(nullptr) {
    data = new float[rows * cols];

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    use_gpu = (err == cudaSuccess && deviceCount > 0);
    if (use_gpu) {
        cudaMalloc(&d_data, rows * cols * sizeof(float));
    }
}

Tensor::~Tensor() {
    delete[] data;
    if (use_gpu) cudaFree(d_data);
}

void Tensor::fill(float value) {
    if (use_gpu) {
        size_t size = rows * cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel << <blocks, threads >> > (d_data, value, size);
        cudaDeviceSynchronize();
        cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void Tensor::randomize() {
    float std_dev = std::sqrt(0.2f / rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
    }

    if (use_gpu) {
        cudaMemcpy(d_data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Tensor::print() const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << " ";
        } 
        std::cout << "\n";
    }
}

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::runtime_error("Matrix size mismatch");
    Tensor C(A.rows, B.cols);

    dim3 threads(16, 16);
    dim3 blocks((C.cols + threads.x - 1) / threads.x, (C.rows + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(A.d_data, B.d_data, C.d_data, A.rows, B.cols, A.cols);
    cudaMemcpy(C.data, C.d_data, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost);
    return C;
}

Tensor Tensor::matadd(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::runtime_error("Matrix size mismatch");
    Tensor C(A.rows, A.cols);
    
    dim3 threads(16, 16);
    dim3 blocks((A.cols + threads.x - 1) / threads.x, (A.rows + threads.y - 1) / threads.y);
    
    matadd_kernel<<<blocks, threads>>>(A.d_data, B.d_data, C.d_data, A.rows, A.cols);
    cudaMemcpy(C.data, C.d_data, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost);
    return C;
}
#endif
