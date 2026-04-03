#pragma once

#include <vector>
#include "tensor.hpp"

namespace bassinet {
    class Layer  {
    public:
        virtual Tensor forward(Tensor& x) = 0;
        virtual std::vector<Tensor*> parameters() = 0;
    };

    class Linear : public Layer {
    private:
        Tensor _weights; // shape: (inputSize, outputSize)
        Tensor _biases; // shape: (outputSize)

    public:
        Linear(std::size_t inputSize, std::size_t outputSize);

        Tensor forward(Tensor& x) override;
        std::vector<Tensor*> parameters() override;
    };
}
