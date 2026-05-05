//
// Created by Linus Bauer on 30.04.26.
//

#include "llm/Linear.h"

Linear::Linear(const size_t inDim, const size_t outDim) : w(inDim, outDim) {

}

Tensor Linear::forward(const Tensor &input) const {
    return Tensor::matmul(input, w);
}

