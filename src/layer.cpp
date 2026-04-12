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


void reluBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child);
bassinet::TensorIntl reluForward(std::shared_ptr<bassinet::TensorIntl> parent) {
    std::shared_ptr<std::vector<float>> resData(std::make_shared<std::vector<float>>(parent->size()));
    for (size_t i = 0; i < resData->size(); ++i) {
        (*resData)[i] = std::max((*parent->data())[i], 0.0f);
    }

    return bassinet::TensorIntl::fromMove(resData,
        parent->shape(), parent->stride(), parent->gradRequired(),
        reluBackward, {parent}
    );
}
void reluBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child) {
    if (parents.size() != 1) throw std::invalid_argument("reluBackward: Operation only supports one parent");

    std::vector<float> grad(child.size());
    for (size_t i = 0; i < child.size(); ++i) {
        grad[i] = ((*parents[0]->data())[i] > 0.0f) ? child.grad()[i] : 0.0f; // note parents are always same size as child for relu
    }

    parents[0]->addToGrad(grad);
}
bassinet::Tensor bassinet::relu(bassinet::Tensor parent) {
    return (std::make_shared<bassinet::TensorIntl>(reluForward(parent.intl)));
}

bassinet::Tensor bassinet::ReLU::forward(Tensor& x) {
    return relu(x);
}
