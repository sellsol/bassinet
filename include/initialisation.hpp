#pragma once

#include <vector>
#include <random>

namespace bassinet {
    void heUniformInit(std::vector<float>& gradients, size_t inputSize, unsigned int seed = 42);
    void heNormalInit(std::vector<float>& gradients, size_t inputSize, unsigned int seed = 42);
}
