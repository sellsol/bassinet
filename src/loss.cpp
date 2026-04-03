#include "loss.hpp"

class MSELoss: public bassinet::TensorOp {
    public:
    bassinet::Tensor forward(const std::vector<bassinet::Tensor*>& parents) override {
        if (parents.size() != 2) throw std::invalid_argument("TensorOop::forward: Operation only supports two parents");
        if (parents[0]->size() != parents[1]->size()) throw std::invalid_argument("mseLoss: prediction and target sizes not equal");

        // TODO: multiple cases at once
        float sum{0};
        for (size_t i = 0; i < parents[0]->size(); ++i) {
            float diff{(*parents[0]->rawData())[i] - (*parents[1]->rawData())[i]};
            sum += diff * diff;
        }

        return bassinet::Tensor(std::vector<float>(1, sum / parents[0]->size()), {1}, {1}, true, [this](std::vector<bassinet::Tensor*>& parents, bassinet::Tensor& child) { this->backward(parents, child); }, parents);
    }

    void backward(const std::vector<bassinet::Tensor*>& parents, bassinet::Tensor& child) override {
        if (parents.size() != 2) throw std::invalid_argument("MatmulOp::forward: Operation only supports two parents");

        if (parents[0]->gradRequired()) parents[0]->addToGrad(child.grad());
        if (parents[1]->gradRequired()) parents[1]->addToGrad(child.grad());
    }
};

bassinet::Tensor bassinet::mseLoss(bassinet::Tensor& pred, bassinet::Tensor& target) {
    MSELoss op;
    return op.forward({&pred, &target});
}
