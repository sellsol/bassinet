#pragma once

#include <vector>
#include <random>
#include "tensor.hpp"

namespace bassinet {
    struct Dataset {
        Tensor X;
        Tensor Y;
    };

    Dataset synthAbs(size_t numSamples, unsigned int seed = 42, float noiseStd = 0.0f);
    Dataset synthXor(size_t numSamples, unsigned int seed = 42);
    // TODO:
    // Dataset synthTwoMoons(size_t numSamples);
    // Dataset synthConcentricCircles(size_t numSamples);
}