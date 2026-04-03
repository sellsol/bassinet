#include "layer.hpp"

bassinet::Linear::Linear(size_t inputSize, size_t outputSize) : _weights(std::vector<size_t>{inputSize, outputSize}, 0.0f, true), _biases(std::vector<size_t>{outputSize}, 0.0f, true) {}

bassinet::Tensor bassinet::Linear::forward(Tensor& x) {
    return x.matmul(_weights) + _biases;
}

std::vector<bassinet::Tensor*> bassinet::Linear::parameters() {
    return {&_weights, &_biases};
}