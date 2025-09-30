#pragma once
#include <cmath>

static float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}
