#include <gtest/gtest.h>

#include "loss.hpp"

TEST(MSELossTest, Forward) {
    bassinet::Tensor pred{std::vector<float>{1.0f, 3.0f}};
    bassinet::Tensor target{std::vector<float>{0.0f, 1.0f}};

    bassinet::Tensor loss = bassinet::mseLoss(pred, target);

    // Mean of squared diffs: ((1-0)^2 + (3-1)^2) / 2 = 2.5
    EXPECT_EQ(loss.intl->shape(), std::vector<size_t>({1}));
    EXPECT_FLOAT_EQ(loss.intl->at({0}), 2.5f);
}

TEST(MSELossTest, Backward) {
    bassinet::Tensor pred = bassinet::Tensor::fromMove(std::vector<float>{2.0f, 4.0f}, {2}, {1}, true);
    bassinet::Tensor target{std::vector<float>{1.0f, 5.0f}};

    bassinet::Tensor loss = bassinet::mseLoss(pred, target);
    EXPECT_FLOAT_EQ(loss.intl->at({0}), 1);

    // Diffs/grads: 2-1 = 1, 4-5 = -1, mean of squares = 1
    loss.intl->backward();
    EXPECT_EQ(pred.intl->grad(), std::vector<float>({1.0f, -1.0f})); // Squared diffs: (2-1)^2, (4-5)^2
}