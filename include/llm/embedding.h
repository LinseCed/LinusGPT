#pragma once
#include "llm/vocab.h"
#include "math/tensor.h"

class Embedding {
public:
    Embedding(int vocabSize, int dim);
    Tensor& forward(int* input, size_t inputSize, Vocab& vocab);
private:
    Tensor weight, gradWeight;
    int input[];
};
