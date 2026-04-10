#include <gtest/gtest.h>

#include "tensor.hpp"
#include "layer.hpp"
#include "optimiser.hpp"
#include "loss.hpp"

TEST(LinearIntn, SingleWeightRegression) {
    std::vector<float> X{-2, -1, 0, 1, 2};
    std::vector<float> Y{-1, 1, 3, 5, 7}; // y = 2x + 3
    bassinet::Linear layer(1, 1); // W: (1,1), b: (1)
    float lr = 0.1f;

    for (int epoch = 0; epoch < 10; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            bassinet::Tensor x{X[i]};
            bassinet::Tensor y{Y[i]};

            // Forward
            bassinet::Tensor pred = layer.forward(x);
            bassinet::Tensor loss = bassinet::mseLoss(pred, y);

            // Backward
            loss.intl->backward();

            // Update
            std::vector<bassinet::Tensor> params = layer.parameters();
            bassinet::sgdStep(params, lr);
        }
    }

    EXPECT_NEAR(layer.parameters()[0].intl->at({0, 0}), 2, 0.05f);
    EXPECT_NEAR(layer.parameters()[1].intl->at({0}), 3, 0.05f);
}

TEST(LinearIntn, TwoWeightsRegression) {
    std::vector<std::vector<float>> X{{1, -2}, {3, 3}, {-2, -1}, {2, 1}, {-1, 2}, {0, 0}};
    std::vector<float> Y{12, 8, 1, 9, -2, 5}; // y = 3x_1 - 2x_2 + 5
    bassinet::Linear layer(2, 1);
    float lr = 0.05f;

    for (int epoch = 0; epoch < 10; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            bassinet::Tensor x{X[i]};
            bassinet::Tensor y{std::vector<float>{Y[i]}};

            // Forward
            bassinet::Tensor pred = layer.forward(x);
            bassinet::Tensor loss = bassinet::mseLoss(pred, y);

            // Backward
            loss.intl->backward();

            // Update
            std::vector<bassinet::Tensor> params = layer.parameters();
            bassinet::sgdStep(params, lr);
        }
    }

    EXPECT_NEAR(layer.parameters()[0].intl->at({0, 0}), 3, 0.05f);
    EXPECT_NEAR(layer.parameters()[0].intl->at({0, 1}), -2, 0.05f);
    EXPECT_NEAR(layer.parameters()[1].intl->at({0}), 5, 0.05f);
}

TEST(LinearIntn, TwoWeightsMultiRegression) {
    std::vector<std::vector<float>> X{{1, -2}, {3, 3}, {-2, -1}, {2, 1}, {-1, 2}, {0, 0}};
    std::vector<std::vector<float>> Y{{5, -13}, {4, 1}, {-2, 0}, {4, -4}, {-3, 9}, {1, -2}}; // y = 2x_1 - x_2 + 1, y = -3x_1 + 4x_2 - 2
    bassinet::Linear layer(2, 2);
    float lr = 0.05f;

    for (int epoch = 0; epoch < 10; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            bassinet::Tensor x{std::vector<float>{X[i]}};
            bassinet::Tensor y{std::vector<float>{Y[i]}};

            // Forward
            bassinet::Tensor pred = layer.forward(x);
            bassinet::Tensor loss = bassinet::mseLoss(pred, y);

            // Backward
            loss.intl->backward();

            // Update
            std::vector<bassinet::Tensor> params = layer.parameters();
            bassinet::sgdStep(params, lr);
        }
    }

    EXPECT_NEAR(layer.parameters()[0].intl->at({0, 0}), 2, 0.1f);
    EXPECT_NEAR(layer.parameters()[0].intl->at({0, 1}), -3, 0.1f);
    EXPECT_NEAR(layer.parameters()[0].intl->at({1, 0}), -1, 0.1f);
    EXPECT_NEAR(layer.parameters()[0].intl->at({1, 1}), 4, 0.1f);
    EXPECT_NEAR(layer.parameters()[1].intl->at({0}), 1, 0.1f);
    EXPECT_NEAR(layer.parameters()[1].intl->at({1}), -2, 0.1f);
}
