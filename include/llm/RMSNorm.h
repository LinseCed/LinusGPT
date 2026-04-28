//
// Created by Linus Bauer on 28.04.26.
//

#ifndef LLM_RMSNORM_H
#define LLM_RMSNORM_H
#include "math/tensor.h"


class RMSNorm {
public:
    explicit RMSNorm(int dim);
    [[nodiscard]] Tensor forward(const Tensor& x) const;
private:
    Tensor gamma, gradientGamma;
    float epsilon;
};


#endif //LLM_RMSNORM_H