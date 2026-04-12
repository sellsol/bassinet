#include "initialisation.hpp"

void bassinet::heUniformInit(std::vector<float>& gradients, size_t inputSize, unsigned int seed) {
    std::mt19937 rng(seed);
    float bound{std::sqrt(6.0f / inputSize)};
    std::uniform_real_distribution<float> uniDist(-bound, bound);
    for (auto it = gradients.begin(); it != gradients.end(); ++it) {
        *it = uniDist(rng);
    }
}

void bassinet::heNormalInit(std::vector<float>& gradients, size_t inputSize, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussDist(0, std::sqrt(2.0f / inputSize));
    for (auto it = gradients.begin(); it != gradients.end(); ++it) {
        *it = gaussDist(rng);
    }
}
