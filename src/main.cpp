#include "tensor.h"
#include <chrono>

int main() {
    const int N = 2048;

    Tensor A(N, N);
    Tensor B(N, N);
    A.randomize();
    B.randomize();
    std::cout << "Running matmul on " << N << "x" << N << " matrices\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    Tensor C_cpu = Tensor::matmul_cpu(A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "CPU time: " << cpu_time << "s\n";

    if (A.rows == B.rows && A.use_gpu) {
        auto g1 = std::chrono::high_resolution_clock::now();
        Tensor C_gpu = Tensor::matmul_gpu(A, B);
        auto g2 = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double>(g2 - g1).count();
        std::cout << "GPU time: " << gpu_time << "s\n";
    } else {
        std::cout << "No GPU available, skipped GPU test.\n";
    }
    return 0;
}
