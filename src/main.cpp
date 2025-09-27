#include "math/tensor.h"
#include <chrono>

int main() {
    const int N = 10;

    Tensor A(N, N);
    Tensor B(N, N);
    A.randomize();
    B.randomize();
    std::cout << "Running matmul on " << N << "x" << N << " matrices\n";
    Tensor C = Tensor::matadd(A, B);
    return 0;
}
