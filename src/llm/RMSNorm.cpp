//
// Created by Linus Bauer on 28.04.26.
//

#include "llm/RMSNorm.h"

RMSNorm::RMSNorm(const int dim) : gamma(dim, 1), gradientGamma(dim, 1), epsilon(1e-5) {
    gamma.fill(1);
    gradientGamma.fill(0);
}

Tensor RMSNorm::forward(const Tensor& x) const {
    const size_t n = x.getRows();
    const size_t d = x.getCols();
    Tensor out(n, d);

    for (size_t i = 0; i < n; i++) {
        auto row = x.getRow(i);
        float mean = 0.0;
        for (const float v : row) {
            mean += v * v;
        }
        mean /= static_cast<float>(d);
        const float rms = std::sqrt(mean + epsilon);

        std::vector<float> result(d);
        for (size_t j = 0; j < d; j++) {
            result[j] = gamma.get(j, 0) * row[j] / rms;
        }
        out.setRow(i, result);
    }
    return out;
}
