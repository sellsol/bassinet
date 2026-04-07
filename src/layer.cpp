#include "layer.hpp"

bassinet::Linear::Linear(size_t inputSize, size_t outputSize):
    _weights(bassinet::Tensor::zeros(std::vector<size_t>{inputSize, outputSize}, true)),
    _biases(bassinet::Tensor::zeros(std::vector<size_t>{outputSize}, true)) {}

bassinet::Tensor bassinet::Linear::forward(Tensor& x) {
    return x.matmul(_weights) + _biases;
}

std::vector<bassinet::Tensor> bassinet::Linear::parameters() {
    return {_weights, _biases};
}