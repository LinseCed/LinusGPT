#include "math/tensor.h"
#include <random>
#include <thread>

#ifndef USE_CUDA
Tensor::Tensor() : Tensor(0, 0) {}

Tensor::Tensor(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
    data = new float[rows * cols];
}

Tensor::~Tensor() {
    delete[] data;
}

Tensor::Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
    if(other.data != nullptr) {
        throw std::runtime_error("Tensor::Tensor(): other.data is null");
    }
    data = new float[rows * cols];
    std::copy_n(other.data, rows * cols, data);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    delete[] data;

    rows = other.rows;
    cols = other.cols;

    data = new float[rows * cols];
    std::copy_n(other.data, rows * cols, data);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        rows = other.rows;
        cols = other.cols;
        data = other.data;

        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

void Tensor::fill(const float value) const {
    std::fill_n(data, rows * cols, value);
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
    if (A.cols != B.rows) throw std::runtime_error("Matrix size mismatch");

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
    size_t rows_per_thread = (A.rows + num_threads - 1)/ num_threads;
    
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
    if (A.rows != B.rows || A.cols != B.cols) throw std::runtime_error("Matrix size mismatch");
    Tensor C(A.rows, A.cols);
    auto worker = [&](const size_t start, const size_t end) {
        for (size_t i = start; i < end; i++)
            C.data[i] = A.data[i] + B.data[i];
    };
    size_t total = A.rows * A.cols;
    size_t num_threads = std::thread::hardware_concurrency();
    num_threads = num_threads == 0 ? 4 : num_threads;
    size_t chunk = (total + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, total);
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();
    return C;
}

void Tensor::setRow(const size_t row, const std::vector<float>& values) const {
	if (rows >= row) throw std::out_of_range("Row index out of range");
    if (values.size() != cols) throw std::runtime_error("Row size mismatch");
    std::ranges::copy(values, data + row * cols);
}

float Tensor::get(const size_t row, const size_t col) const {
    if (row >= rows) throw std::out_of_range("Row index out of range");
    if (col >= cols) throw std::out_of_range("Col index out of range");
    return data[row * cols + col];
}

#endif
