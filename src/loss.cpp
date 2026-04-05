#include "loss.hpp"

// TODO: multiple reduction types - currently only sum
void mselossBackwardSum(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child);
bassinet::Tensor mselossForwardSum(const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) {
    if (parents.size() != 2) throw std::invalid_argument("mselossForward: Operation only supports two parents");
    if (parents[0]->size() != parents[1]->size()) throw std::invalid_argument("mseLoss: prediction and target sizes not equal");

    float sum{0};
    for (size_t i = 0; i < parents[0]->size(); ++i) {
        float diff{(*parents[0]->data())[i] - (*parents[1]->data())[i]};
        sum += diff * diff;
    }

    return bassinet::Tensor(
        std::vector<float>(1, sum / parents[0]->size()), {1}, {1},
        parents[0]->gradRequired() || parents[1]->gradRequired(),
        mselossBackwardSum, parents
    );
}

void mselossBackwardSum(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child) {
    if (parents.size() != 2) throw std::invalid_argument("mselossBackward: Operation only supports two parents");
    if (!parents[0]->gradRequired() || parents[1]->gradRequired()) throw std::invalid_argument("mselossBackward: parents should be in order {pred, target}");

    float sum{child.grad()[0]};
    std::vector<float> predGrad(parents[0]->size());
    for (size_t i = 0; i < predGrad.size(); ++i) {
        float diff{(*parents[0]->data())[i] - (*parents[1]->data())[i]};
        predGrad[i] = sum * (2.0 / predGrad.size()) * diff;
    }

    parents[0]->addToGrad(predGrad);
}

bassinet::Tensor bassinet::mseLoss(bassinet::Tensor& pred, bassinet::Tensor& target) {
    return mselossForwardSum({pred.intl, target.intl}); // order is important for grad func
}
