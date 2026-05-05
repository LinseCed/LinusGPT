#pragma once
#include "math/tensor.h"

class MultiHeadAttention {
public:
    explicit MultiHeadAttention(size_t dim, size_t numHeads);
    [[nodiscard]] Tensor forward(const Tensor &input) const;
private:
    size_t dim, numHeads, headDim;
    Tensor wQ, wK, wV, wO;
};
