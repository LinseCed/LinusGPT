#include "math/tensor.h"
#include <chrono>
#include "llm/vocab.h"

int main() {
    const int N = 2048;
    std::cout << "Loading Vocab\n";
    Vocab vocab = Vocab::getInstance();
    std::cout << vocab.getVocabSize() << "\n";
    Tensor A(N, N);
    Tensor B(N, N);
    A.randomize();
    B.randomize();
    std::cout << "Running matmul on " << N << "x" << N << " matrices\n";
    Tensor C = Tensor::matmul(A, B);
    return 0;
}
