#pragma once
#include <cmath>
#include "math/tensor.h"

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

Tensor softmax(Tensor logits) {
    Tensor result(logits.getRows(), logits.getCols());
    int rows = logits.getRows();
    for (int i = 0; i < rows; i++) {
        int n = logits.getCols();
        float max = -1 * INFINITY;
        for (int j = 0; j < n; j++) {
            if (logits.)
        }
    }
}
