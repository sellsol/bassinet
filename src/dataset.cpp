#include "dataset.hpp"

bassinet::Dataset bassinet::synthAbs(size_t numSamples, unsigned int seed, float noiseStd) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniDist(-2.0f, 2.0f);
    std::normal_distribution<float> gaussDist(0.0f, noiseStd);

    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> Y;
    for (size_t i = 0; i < numSamples; ++i) {
        float x{uniDist(rng)};
        X.push_back({x + gaussDist(rng)});
        Y.push_back({std::abs(x)});
    }

    return {bassinet::Tensor{X}, bassinet::Tensor{Y}};
}

bassinet::Dataset bassinet::synthXor(size_t numSamples, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-5, 5);

    std::vector<std::vector<std::vector<float>>> X;
    std::vector<std::vector<float>> Y;
    for (size_t i = 0; i < numSamples; ++i) {
        std::vector<float> x{distribution(rng), distribution(rng)};
        X.push_back({x});

        if ((x[0] >= 0 && x[1] >= 0) || (x[0] < 0 && x[1] < 0)) Y.push_back({0}); // for simplicity let 0,0 be 0 also
        else Y.push_back({1});
    }

    return {bassinet::Tensor{X}, bassinet::Tensor{Y}};
}