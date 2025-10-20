#pragma once
#include "llm/vocab.h"
#include "math/tensor.h"

class Embedding {
public:
    Embedding(int vocabSize, int dim);
    Tensor forward(std::vector<int> input, Vocab& vocab);
private:
    Tensor weight, gradWeight;
    std::vector<int> input;
};
