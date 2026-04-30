#include "math/tensor.h"
#include <random>
#include <thread>

#ifndef USE_CUDA

constexpr size_t TILE = 32;

Tensor::Tensor() : Tensor(0, 0) {}

Tensor::Tensor(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
    data = (rows * cols > 0) ?  std::make_unique<float[]>(rows * cols) : nullptr;
}

Tensor::~Tensor() = default;

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
    if(other.data == nullptr) {
        data = nullptr;
        return;
    }
    data = std::make_unique<float[]>(rows * cols);
    std::copy_n(other.data.get(), rows * cols, data.get());
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    rows = other.rows;
    cols = other.cols;

    data = std::make_unique<float[]>(rows * cols);
    std::copy_n(other.data.get(), rows * cols, data.get());
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data.get()), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    rows = other.rows;
    cols = other.cols;
    if (other.data == nullptr) {
        data = nullptr;
        return *this;
    }
    data = std::make_unique<float[]>(rows * cols);
    std::copy_n(other.data.get(), rows * cols, data.get());
    return *this;
}

void Tensor::fill(const float value) const {
    std::fill_n(data.get(), rows * cols, value);
}

void Tensor::randomize() const {
    const float std_dev = std::sqrt(0.2f / static_cast<float>(rows));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < rows * cols; i++) data[i] = dist(gen);
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
#ifndef NDEBUG
    if (A.cols != B.rows) throw std::runtime_error("Matrix size mismatch");
#endif
    Tensor C(A.rows, B.cols);
    auto worker = [&](const size_t start_row, const size_t end_row) {
        for (size_t i = start_row; i < end_row; i++) {
            for (size_t j = 0; j < B.cols; j++) {
                float sum = 0.0;
                for (size_t k = 0; k < A.cols; k++) {
                    sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
                C.data[i * B.cols + j] = sum;
            }
        }
    };

    size_t num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 4 : num_threads;
    const size_t rows_per_thread = (A.rows + num_threads - 1)/ num_threads;
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * rows_per_thread;
        size_t end = std::min(start + rows_per_thread, A.rows);
        if (start < end)
            threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();
    return C;
}

Tensor Tensor::matadd(const Tensor& A, const Tensor& B) {
#ifndef NDEBUG
    if (A.rows != B.rows || A.cols != B.cols) throw std::runtime_error("Matrix size mismatch");
#endif
    Tensor C(A.rows, A.cols);
    auto worker = [&](const size_t start, const size_t end) {
        for (size_t i = start; i < end; i++)
            C.data[i] = A.data[i] + B.data[i];
    };
    const size_t total = A.rows * A.cols;
    size_t num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 4 : num_threads;
    const size_t chunk = (total + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk;
        if (size_t end = std::min(start + chunk, total); start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();
    return C;
}

Tensor Tensor::transpose() const {
    Tensor result(cols, rows);
    for (size_t i = 0; i < rows; i += TILE) {
        for (size_t j = 0; j < cols; j += TILE) {
            for (size_t ii = i; ii < std::min(i + TILE, rows); ii++) {
                for (size_t jj = j; jj < std::min(j + TILE, cols); jj++) {
                    result.set(jj, ii, data[ii * cols + jj]);
                }
            }
        }
    }
    return result;
}

Tensor Tensor::slice(const size_t colStart, const size_t colEnd) const {
    Tensor result(rows, colEnd - colStart);
    if (result.data == nullptr) return result;
    for (size_t i = 0; i < rows; i++) {
        memcpy(result.data.get() + i * result.cols, data.get() + i * cols + colStart, result.cols * sizeof(float));
    }
    return result;
}


void Tensor::scale(float scale) const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i * cols + j] *= scale;
        }
    }
}

void Tensor::setRow(const size_t row, const std::vector<float>& values) const {
#ifndef NDEBUG
	if (rows <= row) throw std::out_of_range("Row index out of range");
    if (values.size() != cols) throw std::runtime_error("Row size mismatch");
#endif
    std::ranges::copy(values, data.get() + row * cols);
}

std::vector<float> Tensor::getRow(const size_t row) const {
    return {data.get() + row * cols, data.get() + (row + 1) * cols};
}


float Tensor::get(const size_t row, const size_t col) const {
#ifndef NDEBUG
    if (row >= rows) throw std::out_of_range("Row index out of range");
    if (col >= cols) throw std::out_of_range("Col index out of range");
#endif
    return data[row * cols + col];
}

void Tensor::set(const size_t row, const size_t col, const float value) const {
    data[row * cols + col] = value;
}


#endif
