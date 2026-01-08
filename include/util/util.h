#pragma once
#include <cmath>
#include "math/tensor.h"

inline float sigmoid(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline Tensor softmax(const Tensor& logits) {
    Tensor result(logits.getRows(), logits.getCols());
    const size_t rows = logits.getRows();
    for (size_t i = 0; i < rows; i++) {
        const size_t n = logits.getCols();
        float max = -1 * INFINITY;
        for (size_t j = 0; j < n; j++) {
            if (logits.get(i, j) > max) {
                max = logits.get(i, j);
            }
        }
        const auto expVals = new float[n];
        float sum = 0;
        for (size_t j = 0; j < n; j++) {
            expVals[j] = exp(logits.get(i, j) - max);
            sum += expVals[j];
        }

        for (size_t j = 0; j < n; j++) {
            expVals[j] /= sum;
        }
        result.setRow(i, std::vector<float>(expVals, expVals + n));
    }
    return result;
}
