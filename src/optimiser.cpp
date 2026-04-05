#include "optimiser.hpp"

void bassinet::sgdStep(std::vector<bassinet::Tensor>& params, float lr) {
    for (Tensor p : params) {
        if (!p.intl->gradRequired()) continue;

        std::vector<float> additions(p.intl->size());
        for (size_t i = 0; i < p.intl->size(); ++i) {
            additions[i] -= lr * (p.intl->grad())[i];
        }
        p.intl->addToData(additions);
        p.intl->zeroGrad();
    }
}
