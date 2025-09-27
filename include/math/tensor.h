#pragma once
#include <cstddef>
#include <vector>
#include <iostream>

class Tensor {
public:
    Tensor(size_t rows, size_t cols);
    ~Tensor();

    void fill(float value);
    void randomize();
    void print() const;
    
    static Tensor matmul(const Tensor& A, const Tensor& B);
    static Tensor matadd(const Tensor& A, const Tensor& B);

    size_t rows, cols;
    bool use_gpu;
private:
    float* data;
    float* d_data;

    void fill_cpu(float value);
    
    static Tensor matmul_cpu(const Tensor& A, const Tensor& B);
    static Tensor matmul_gpu(const Tensor& A, const Tensor& B);
    static Tensor matadd_cpu(const Tensor& A, const Tensor& B);
    static Tensor matadd_gpu(const Tensor& A, const Tensor& B);
};
