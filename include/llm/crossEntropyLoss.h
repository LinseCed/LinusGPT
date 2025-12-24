#pragma once

#import "math/tensor.h"

double forward(Tensor logits, std::vector<int> target) {
    double loss = 0;
    Tensor softmaxedLogits();
}
