#pragma once
#include "math/tensor.h"
class LLM {
public:
    LLM(int vocabSize, int dim, int numBlocks, int numHeads, int ffDim);

    Tensor forward(int[]);
};
