#include "llm/Embedding.h"

Embedding::Embedding(const int vocabSize, const int dim, const int contextLength) : tokenEmbedding(vocabSize, dim), gradientToken(vocabSize, dim), positionEmbedding(contextLength, dim), gradientPosition(contextLength, dim), dim(dim), contextLength(contextLength) {
	tokenEmbedding.randomize();
	gradientToken.fill(0);
	positionEmbedding.randomize();
	gradientPosition.fill(0);
}

Tensor Embedding::forward(const std::vector<int>& inputTokens) {
	this->input = inputTokens;
	const size_t n = inputTokens.size();
	const Tensor tok(n, dim);
	const Tensor pos(n, dim);
	for (size_t i = 0; i < n; i++) {
		tok.setRow(i, tokenEmbedding.getRow(inputTokens[i]));
		pos.setRow(i, positionEmbedding.getRow(i));
	}
	return Tensor::matadd(tok, pos);
}