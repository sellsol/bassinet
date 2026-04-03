#include "optimiser.hpp"

void bassinet::sgdStep(std::vector<bassinet::Tensor*>& params, float lr) {
    for (Tensor* p : params) {
        if (!p->gradRequired()) continue;

        std::vector<float> additions(p->size());
        for (size_t i = 0; i < p->size(); ++i) {
            additions[i] -= lr * (p->grad())[i];
        }
        p->addToData(additions);
        p->zeroGrad();
    }
}
