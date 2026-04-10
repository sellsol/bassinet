#include <gtest/gtest.h>

#include "optimiser.hpp"

TEST(SGDStepTest, Forward) {
    bassinet::Tensor param = bassinet::Tensor::zeros({2}, true);
    param.intl->addToGrad({2.0f, -4.0f});
    std::vector<bassinet::Tensor> params{param};

    bassinet::sgdStep(params, 0.1f); // values should change by 0.1 of gradient vals {2, -4}
    EXPECT_FLOAT_EQ(param.intl->at({0}), -0.2f);
    EXPECT_FLOAT_EQ(param.intl->at({1}), 0.4f);
    EXPECT_EQ(param.intl->grad(), std::vector<float>({0.0f, 0.0f})); // gradients are zeroes in prep for next step
}

TEST(SGDStepTest, ForwardNotRequiredGrad) {
    bassinet::Tensor param = bassinet::Tensor::fromMove(std::vector<float>{3.0f, -1.0f}, {2}, {1}, false);
    std::vector<bassinet::Tensor> params{param};

    bassinet::sgdStep(params, 0.1f);
    EXPECT_FLOAT_EQ(param.intl->at({0}), 3.0f);
    EXPECT_FLOAT_EQ(param.intl->at({1}), -1.0f);
}