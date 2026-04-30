#pragma once
#include "llm/Vocab.h"
#include "math/tensor.h"

class Embedding {
public:
    Embedding(int vocabSize, int dim, int contextLength);
    Tensor forward(const std::vector<int>& inputTokens);
private:
    Tensor tokenEmbedding, gradientToken;
    Tensor positionEmbedding, gradientPosition;
    std::vector<int> input;
    int dim, contextLength;
};
