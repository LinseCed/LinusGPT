#include "llm/embedding.h"

Embedding::Embedding(int vocabSize, int dim) : weight(vocabSize, dim), gradWeight(vocabSize, dim) {
	weight.randomize();
	gradWeight.fill(0);
}

Tensor& Embedding::forward(int* input, size_t inputSize, Vocab& vocab) {
	weight.print();
	return weight;
}