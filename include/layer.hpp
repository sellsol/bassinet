#pragma once

#include <vector>
#include "tensor.hpp"

namespace bassinet {
    class Linear {
    private:
        Tensor _weights; // shape: (inputSize, outputSize)
        Tensor _biases; // shape: (outputSize)

    public:
        Linear(size_t inputSize, size_t outputSize);

        Tensor forward(Tensor& x);
        std::vector<Tensor> parameters();
    };
}
