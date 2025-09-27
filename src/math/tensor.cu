#include "math/tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <chrono>

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

void Tensor::fill_cpu(float value) {
    for (size_t i = 0; i < rows * cols; i++) data[i] = value;
}

void Tensor::fill(float value) {
    if (use_gpu) {
        size_t size = rows * cols;
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(d_data, value, size);
        cudaDeviceSynchronize();
        cudaMemcpy(data, d_data, size*sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        fill_cpu(value);
    }
}

void Tensor::randomize() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(rng);
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

Tensor Tensor::matmul_cpu(const Tensor& A, const Tensor& B) {
    Tensor C(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; i++) {
        for (size_t j = 0; j < B.cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            C.data[i * B.cols + j] = sum;
        }
    }
    return C;
}

Tensor Tensor::matmul_gpu(const Tensor& A, const Tensor& B) {
    Tensor C(A.rows, B.cols);

    dim3 threads(16, 16);
    dim3 blocks((B.cols + threads.x - 1) / threads.x, (A.rows + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(A.d_data, B.d_data, C.d_data, A.rows, B.cols, A.cols);
    cudaMemcpy(C.data, C.d_data, A.rows * B.cols * sizeof(float), cudaMemcpyDeviceToHost);
    return C;
}

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::runtime_error("Matrix size mismatch");
    if (A.use_gpu && B.use_gpu) {
        return matmul_gpu(A, B);
    } else {
        return matmul_cpu(A, B);
    }
}
