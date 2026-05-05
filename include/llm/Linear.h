//
// Created by Linus Bauer on 30.04.26.
//
#pragma once

#include "math/tensor.h"

class Linear {
public:
    Linear(size_t inDim, size_t outDim);
    [[nodiscard]] Tensor forward(const Tensor& input) const;
private:
    Tensor w, b;
};
