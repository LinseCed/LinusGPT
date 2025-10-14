#pragma once
#include "llm/vocab.h"
#include "math/tensor.h"

class Embedding {
public:
    Embedding(Vocab& vocab);
    Tensor& forward(int* input, size_t inputSize, Vocab& vocab);
private:
   Tensor weight, gradWeight;
   int input[];
};
