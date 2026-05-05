//
// Created by Linus Bauer on 30.04.26.
//

#include "llm/MultiHeadAttention.h"

#include "util/util.h"

MultiHeadAttention::MultiHeadAttention(size_t dim, size_t numHeads) : dim(dim), numHeads(numHeads), headDim(dim / numHeads), wQ(dim, dim), wK(dim, dim), wV(dim, dim), wO(dim, dim) {
    if (dim % numHeads != 0) {
        throw std::invalid_argument("dim must be divisible by numHeads");
    }
    wQ.randomize(), wK.randomize(), wV.randomize(), wO.randomize();
}

Tensor MultiHeadAttention::forward(const Tensor &input) const {
    const size_t seq = input.getRows();

    const Tensor Q = Tensor::matmul(input, wQ);
    const Tensor K = Tensor::matmul(input, wK);
    const Tensor V = Tensor::matmul(input, wV);

    const Tensor concat(seq, dim);
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    for (size_t h = 0; h < numHeads; h++) {
        const size_t colStart = h * headDim;
        const size_t colEnd = colStart + headDim;

        Tensor Qh = Q.slice(colStart, colEnd);
        Tensor Kh = K.slice(colEnd, colStart);
        Tensor Vh = V.slice(colStart, colEnd);

        Tensor scores = Tensor::matmul(Qh, Kh.transpose());
        scores.scale(scale);

        Tensor attn = softmax(scores);
        Tensor headOut = Tensor::matmul(attn, Vh);

        for (size_t i = 0; i < seq; i++) {
            for (size_t j = 0; j < headDim; j++) {
                concat.set(i, colStart + j, headOut.get(i, j));
            }
        }
    }
    return Tensor::matmul(concat, wO);
}
