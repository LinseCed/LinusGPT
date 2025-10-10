#include "math/tensor.h"
#include <random>
#include <thread>

Tensor::Tensor(size_t rows, size_t cols) : rows(rows), cols(cols), d_data(nullptr) {
    data = new float[rows * cols];
}

Tensor::~Tensor() {
    delete[] data;
}

void Tensor::fill(float value) {
    for (size_t i = 0; i < rows * cols; i++) data[i] = value;
}

void Tensor::randomize() {
    float std_dev = std::sqrt(0.2f / rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (size_t i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
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
    Tensor C(A.rows, B.cols);
    auto worker = [&](size_t start_row, size_t end_row) {
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
    size_t rows_per_thread = A.rows / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * rows_per_thread;
        size_t end = (t == num_threads - 1) ? A.rows : start + rows_per_thread;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();
    return C;
}


Tensor Tensor::matadd(const Tensor& A, const Tensor& B) {
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.rows; i++) {
        for (size_t j = 0; j < A.cols; j++) {
            C.data[i * A.cols + j] = A.data[i * A.cols + j] + B.data[i * A.cols + j];
        }
    }
    return C;
}
