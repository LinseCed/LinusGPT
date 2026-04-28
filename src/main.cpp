#include "math/tensor.h"

#include "llm/embedding.h"
#include "llm/vocab.h"

int main() {
  const int N = 5000;
  std::cout << "Loading Vocab\n";
  const Vocab vocab = Vocab::getInstance();
  auto inputTokens = vocab.encode("H");
  std::cout << vocab.getVocabSize() << "\n";
  Embedding embedding{vocab.getVocabSize(), 12, 128};
  auto out = embedding.forward(inputTokens);
  out.print();
  Tensor A(N, N);
  Tensor B(N, N);
  A.randomize();
  B.randomize();
  std::cout << "Running matmul on " << N << "x" << N << " matrices\n";
  Tensor C = Tensor::matmul(A, B);
  return 0;
}
