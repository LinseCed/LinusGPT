#include "math/tensor.h"
#include <chrono>

int main() {
    const int N = 2048;

    Tensor A(N, N);
    Tensor B(N, N);
    A.randomize();
    B.randomize();
    std::cout << "Running matmul on " << N << "x" << N << " matrices\n";
    Tensor C = Tensor::matmul(A, B);
    return 0;
}
