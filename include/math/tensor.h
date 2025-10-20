#pragma once
#include <cstddef>
#include <vector>
#include <iostream>

class Tensor {
public:
    Tensor();
    Tensor(size_t rows, size_t cols);
    ~Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    float* data = nullptr;
#ifdef USE_CUDA
    float* d_data = nullptr;
#endif

    void fill(float value);
    void randomize();
    void print() const;
    
    static Tensor matmul(const Tensor& A, const Tensor& B);
    static Tensor matadd(const Tensor& A, const Tensor& B);

    size_t rows, cols;
};
