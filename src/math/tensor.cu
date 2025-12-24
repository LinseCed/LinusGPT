#include "math/tensor.h"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#ifdef USE_CUDA

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

Tensor::Tensor(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data = new float[rows * cols];

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
        cudaMalloc(&d_data, rows * cols * sizeof(float));
    }
}

Tensor::~Tensor() {
    delete[] data;
    if (d_data) cudaFree(d_data);
}

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
    data = new float[rows * cols];
    std::copy(other.data, other.data + rows * cols, data);
    
    d_data = nullptr;
    if (other.data) {
        cudaMalloc(&d_data, rows * cols * sizeof(float));
        cudaMemcpy(d_data, other.d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    delete[] data;
    if (d_data) cudaFree(d_data);

    rows = other.rows;
    cols = other.cols;

    data = new float[rows * cols];
    std::copy(other.data, other.data + rows * cols, data);

    d_data = nullptr;
    if (other.d_data) {
        cudaMalloc(&d_data, rows * cols * sizeof(float));
        cudaMemcpy(d_data, other.d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data), d_data(other.d_data) {
    other.data = nullptr;
    other.d_data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        if (d_data) cudaFree(d_data);

        rows = other.rows;
        cols = other.cols;
        data = other.data;
        d_data = other.d_data;

        other.data = nullptr;
        other.d_data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    
    return *this;
}

void Tensor::fill(float value) {
    size_t size = rows * cols;
    if (d_data) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(d_data, value, size);
        cudaDeviceSynchronize();
        cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else {
        std::fill(data, data + size, value);
    }
}

void Tensor::randomize() {
    float std_dev = std::sqrt(0.2f / rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < rows * cols; i++) data[i] = dist(gen);
    
    if (d_data) cudaMemcpy(d_data, data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
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

    if (A.d_data && B.d_data) {
        dim3 threads(16, 16);
        dim3 blocks((C.cols + threads.x - 1) / threads.x, (C.rows + threads.y - 1) / threads.y);
        matmul_kernel << <blocks, threads >> > (A.d_data, B.d_data, C.d_data, A.rows, B.cols, A.cols);
        cudaMemcpy(C.data, C.d_data, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else {
        for (size_t i = 0; i < A.rows; i++) {
            for (size_t j = 0; j < B.cols; j++) {
                float sum = 0;
                for (size_t k = 0; k < A.cols; k++)
                    sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                C.data[i * B.cols + j] = sum;
            }
        }
    }
    return C;
}

Tensor Tensor::matadd(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::runtime_error("Matrix size mismatch");

    Tensor C(A.rows, A.cols);
    if (A.d_data && B.d_data) {
        dim3 threads(16, 16);
        dim3 blocks((A.cols + threads.x - 1) / threads.x, (A.rows + threads.y - 1) / threads.y);
        matadd_kernel << <blocks, threads >> > (A.d_data, B.d_data, C.d_data, A.rows, A.cols);
        cudaMemcpy(C.data, C.d_data, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost);
    }
    else {
        for (size_t i = 0; i < A.rows * A.cols; i++) C.data[i] = A.data[i] + B.data[i];
    }
    return C;
}

void Tensor::setRow(int row, std::vector<float>& values) {
	if (row >= rows) throw std::out_of_range("Row index out of range");
	if (values.size() != cols) throw std::invalid_argument("Values size mismatch");
	float* dest = data + row * cols;
	cudaError_t err = cudaMemcpy(dest, values.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA memcpy failed") + cudaGetErrorString(err));
}

float Tensor::get(size_t row, size_t col) const {
    if (row >= rows) throw std::out_of_range("Row index out of range");
    if (col >= cols) throw std::out_of_range("Col index out of range");
    return data[row * cols + col];
}
#endif
