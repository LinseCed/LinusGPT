#pragma once
#include <cstddef>
#include <vector>
#include <iostream>

class Tensor {
public:
    Tensor(size_t rows, size_t cols);
    ~Tensor();

    void fill(float value);
    void print() const;

private:
    size_t rows, cols;
    float* data;
    float* d_data;
    bool use_gpu;

    void fill_cpu(float value);
};
