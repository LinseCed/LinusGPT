#pragma once
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

    void fill(float value) const;
    void randomize() const;
    void print() const;
    
    static Tensor matmul(const Tensor& A, const Tensor& B);
    static Tensor matadd(const Tensor& A, const Tensor& B);
    static Tensor transpose(const Tensor& A);

    void scale(const Tensor& A, float scale) const;
    void setRow(size_t row, const std::vector<float>& values) const;
    [[nodiscard]] std::vector<float> getRow(size_t row) const;

    [[nodiscard]] float get(size_t row, size_t col) const;
    void set(size_t row, size_t col, float value) const;

	[[nodiscard]] size_t getRows() const { return rows; }
	[[nodiscard]] size_t getCols() const { return cols; }
private:
    size_t rows, cols;
};
