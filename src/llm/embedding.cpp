#include "llm/embedding.h"

Embedding::Embedding(int vocabSize, int dim) : weight(vocabSize, dim), gradWeight(vocabSize, dim) {
	weight.randomize();
	gradWeight.fill(0);
}

Tensor Embedding::forward(std::vector<int> input, Vocab& vocab) {
	this->input = input;
	int n = input.size();
	Tensor out(n, weight.getCols());
	for (int i = 0; i < n; i++) {
		// TODO setRow function on Tensor
	}
	return out;
}