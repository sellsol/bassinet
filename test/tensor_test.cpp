#include <gtest/gtest.h>

#include "tensor.hpp"

TEST(TensorIntlTest, ConstructZeros) {
    bassinet::TensorIntl t(bassinet::TensorIntl::zeros({2, 3}));
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
}

TEST(TensorIntlTest, ConstructFull) {
    bassinet::TensorIntl t(bassinet::TensorIntl::full({2, 3}, 4.0f));
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 4.0f);
}

TEST(TensorIntlTest, ConstructInitlist2D) {
    bassinet::TensorIntl t{
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}
TEST(TensorIntlTest, ConstructInitlist3D) {
    bassinet::TensorIntl t{{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    },
    {
        {5.0f, 6.0f},
        {7.0f, 8.0f}
    }};
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 2, 2}));
    EXPECT_EQ(t.size(), 8);
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1, 1}), 8.0f);
}
TEST(TensorIntlTest, ConstructInitlistScalar) {
    bassinet::TensorIntl t{1.0f};
    EXPECT_EQ(t.shape(), std::vector<size_t>({1}));
    EXPECT_EQ(t.size(), 1);
    EXPECT_FLOAT_EQ(t.at({0}), 1.0f);
}

TEST(TensorIntlTest, ConstructTemplate2D) {
    std::vector<std::vector<float>> data{
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    bassinet::TensorIntl t(data);
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}
TEST(TensorIntlTest, ConstructTemplate3D) {
    std::vector<std::vector<std::vector<float>>> data{{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    },
    {
        {5.0f, 6.0f},
        {7.0f, 8.0f}
    }};
    bassinet::TensorIntl t(data);
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 2, 2}));
    EXPECT_EQ(t.size(), 8);
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1, 1}), 8.0f);
}
TEST(TensorIntlTest, ConstructTemplateRagged) {
    std::vector<std::vector<float>> ragged = {
        {1.0f, 2.0f},
        {3.0f}
    };

    EXPECT_THROW(bassinet::TensorIntl t(ragged), std::invalid_argument);
}

TEST(TensorIntlTest, ConstructFromMoveVector) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    bassinet::TensorIntl t = bassinet::TensorIntl::fromMove(data, {2, 2}, {2, 1}, true);

    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 2}));
    EXPECT_EQ(t.stride(), std::vector<size_t>({2, 1}));
    EXPECT_EQ(t.size(), 4);
    EXPECT_EQ(t.grad(), std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}
TEST(TensorIntlTest, FromMoveSharedptr) {
    auto data = std::make_shared<std::vector<float>>(std::vector<float>{5.0f, 6.0f});
    bassinet::TensorIntl t = bassinet::TensorIntl::fromMove(data, {2}, {1});

    EXPECT_EQ(t.shape(), std::vector<size_t>({2}));
    EXPECT_EQ(t.stride(), std::vector<size_t>({1}));
    EXPECT_EQ(t.data(), data);
    EXPECT_FLOAT_EQ(t.at({0}), 5.0f);
    EXPECT_FLOAT_EQ(t.at({1}), 6.0f);
}
TEST(TensorIntlTest, FromMoveRejectsShapeStrideMismatch) {
    EXPECT_THROW(
        bassinet::TensorIntl::fromMove(std::vector<float>{1.0f, 2.0f}, {2, 2}, {2, 1}),
        std::invalid_argument
    );
}


TEST(TensorIntlTest, At) {
    bassinet::TensorIntl t{{
        {4.2f, 3.3f},
        {2.0f, 2.0f}
    },
    {
        {0.0f, 1.4f},
        {2.0f, 0.0f}
    }};
    EXPECT_FLOAT_EQ(t.at({0, 0, 0}), 4.2f);
    EXPECT_FLOAT_EQ(t.at({1, 0, 1}), 1.4f);
}
TEST(TensorIntlTest, AtModify) {
    bassinet::TensorIntl t(bassinet::TensorIntl::zeros({2, 2}));
    t.at({0, 1}) = 3.14f;
    EXPECT_FLOAT_EQ(t.at({0, 1}), 3.14f);
}

TEST(TensorIntlTest, Transpose2D) {
    bassinet::TensorIntl t{
        {1, 2, 3},
        {4, 5, 6}
    };
    bassinet::TensorIntl tT = *t.transpose(0, 1);
    EXPECT_EQ(tT.shape(), std::vector<size_t>({3, 2}));
    EXPECT_FLOAT_EQ(tT.at({0, 0}), 1);
    EXPECT_FLOAT_EQ(tT.at({1, 0}), 2);
    EXPECT_FLOAT_EQ(tT.at({2, 0}), 3);
    EXPECT_FLOAT_EQ(tT.at({0, 1}), 4);
    EXPECT_FLOAT_EQ(tT.at({1, 1}), 5);
    EXPECT_FLOAT_EQ(tT.at({2, 1}), 6);
}
TEST(TensorIntlTest, Transpose3D) {
    bassinet::TensorIntl t(bassinet::TensorIntl::zeros({2, 3, 4}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                t.at({i, j, k}) = 100 * i + 10 * j + k;

    bassinet::TensorIntl tT = *t.transpose(0, 2);
    EXPECT_EQ(tT.shape(), std::vector<size_t>({4, 3, 2}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(tT.at({k, j, i}), t.at({i, j, k}));
}


TEST(TensorTest, ConstructZeros) {
    bassinet::Tensor t(bassinet::Tensor::zeros({2, 3}));
    EXPECT_EQ(t.intl->shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.intl->size(), 6);
}

TEST(TensorTest, ConstructFrom2D) {
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    bassinet::Tensor t(data);
    EXPECT_EQ(t.intl->shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.intl->size(), 6);
    EXPECT_FLOAT_EQ(t.intl->at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.intl->at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.intl->at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.intl->at({1, 2}), 6.0f);
}

TEST(TensorTest, ConstructFromNestedVector3D) {
    bassinet::Tensor t({
        {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        },
        {
            {5.0f, 6.0f},
            {7.0f, 8.0f}
        }
    });
    EXPECT_EQ(t.intl->shape(), std::vector<size_t>({2, 2, 2}));
    EXPECT_EQ(t.intl->size(), 8);
    EXPECT_FLOAT_EQ(t.intl->at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.intl->at({0, 1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(t.intl->at({1, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(t.intl->at({1, 1, 1}), 8.0f);
}

TEST(TensorTest, ConstructFromScalarInitlist) {
    bassinet::Tensor t{1.0f};
    EXPECT_EQ(t.intl->shape(), std::vector<size_t>({1}));
    EXPECT_EQ(t.intl->size(), 1);
    EXPECT_FLOAT_EQ(t.intl->at({0}), 1.0f);
}

TEST(TensorTest, Addition) {
    bassinet::Tensor a({{1.0f, 1.0f}, {1.0f, 1.0f}});
    bassinet::Tensor b({{2.0f, 2.0f}, {2.0f, 2.0f}});
    bassinet::Tensor c = a + b;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c.intl->at({i, j}), 3.0f);
        }
    }
}
TEST(TensorTest, AdditionBroadcastScalar) {
    bassinet::Tensor a({{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}});
    bassinet::Tensor b({{2.0f, 2.0f, 2.0f}, {2.0f, 2.0f, 2.0f}});
    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.intl->shape(), std::vector<size_t>({2, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(c.intl->at({i, j}), 3.0f);
}
TEST(TensorTest, AdditionBroadcastCol) {
    bassinet::Tensor a = bassinet::Tensor::full({2, 3}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({2, 1}, 2.0f);
    a.intl->at({1, 2}) = 4.0f;

    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.intl->shape(), std::vector<size_t>({2, 3}));
    EXPECT_FLOAT_EQ(c.intl->at({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(c.intl->at({1, 2}), 6.0f);
}
TEST(TensorTest, AdditionBroadcastBoth) {
    bassinet::Tensor a = bassinet::Tensor::full({2, 1, 3}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({1, 4, 1}, 2.0f);
    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.intl->shape(), std::vector<size_t>({2, 4, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 3; ++k)
                EXPECT_FLOAT_EQ(c.intl->at({i, j, k}), 3.0f);
}
TEST(TensorTest, AdditionShapeMismatch) {
    bassinet::Tensor a = bassinet::Tensor::full({2, 2}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({2, 3}, 2.0f);
    EXPECT_THROW(a + b, std::invalid_argument);
}


TEST(TensorTest, Matmul1Dx1D) {
    bassinet::Tensor a{1.0f, 2.0f, 3.0f};
    bassinet::Tensor b{4.0f, 5.0f, 6.0f};
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({1}));
    EXPECT_FLOAT_EQ(result.intl->at({0}), 32);
}
TEST(TensorTest, Matmul2Dx2D) {
    bassinet::Tensor a{
        {1, 2, 3},
        {4, 5, 6}
    };
    bassinet::Tensor b{
        {7, 8},
        {9, 10},
        {11, 12}
    };
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2,2}));
    EXPECT_FLOAT_EQ(result.intl->at({0,0}), 58);
    EXPECT_FLOAT_EQ(result.intl->at({0,1}), 64);
    EXPECT_FLOAT_EQ(result.intl->at({1,0}), 139);
    EXPECT_FLOAT_EQ(result.intl->at({1,1}), 154);
}
TEST(TensorTest, Matmul1Dx2D) {
    bassinet::Tensor a{1, 2, 3};
    bassinet::Tensor b{
        {4, 5},
        {6, 7},
        {8, 9}
    };
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2}));
    EXPECT_FLOAT_EQ(result.intl->at({0}), 40);
    EXPECT_FLOAT_EQ(result.intl->at({1}), 46);
}
TEST(TensorTest, Matmul2Dx1D) {
    bassinet::Tensor a{
        {1, 2, 3},
        {4, 5, 6}
    };
    bassinet::Tensor b{7, 8, 9};
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2}));
    EXPECT_FLOAT_EQ(result.intl->at({0}), 50);
    EXPECT_FLOAT_EQ(result.intl->at({1}), 122);
}
TEST(TensorTest, MatmulBatched3Dx3D) {
    bassinet::Tensor a{{
        {0, 2}, // batch 0
        {-2, -5}
    },
    {
        {-5, -5}, // batch 1
        {-1, 2}
    }};
    bassinet::Tensor b{{
        {6, -6},
        {3, 0}
    },
    {
        {-2, -3},
        {3, 5}
    }};
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2,2,2}));
    EXPECT_FLOAT_EQ(result.intl->at({0,0,0}), 6);
    EXPECT_FLOAT_EQ(result.intl->at({1,1,1}), 13);
}
TEST(TensorTest, MatmulShapeMismatch) {
    bassinet::Tensor a = bassinet::Tensor::full({2, 3}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({4, 2}, 1.0f);
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}
TEST(TensorTest, MatmulBroadcastBatchLeft) {
    bassinet::Tensor a = bassinet::Tensor::full({2, 3, 4}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({4, 5}, 2.0f);
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2, 3, 5}));
    EXPECT_FLOAT_EQ(result.intl->at({0,0,0}), 8.0f);
    EXPECT_FLOAT_EQ(result.intl->at({1,2,4}), 8.0f);
}
TEST(TensorTest, MatmulBroadcastBatchRight) {
    bassinet::Tensor a = bassinet::Tensor::full({4, 3}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({2, 3, 5}, 2.0f);
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({2, 4, 5}));
    EXPECT_FLOAT_EQ(result.intl->at({0,0,0}), 6.0f);
    EXPECT_FLOAT_EQ(result.intl->at({1,3,4}), 6.0f);
}

TEST(TensorTest, MatmulBroadcastBothSides) {
    bassinet::Tensor a = bassinet::Tensor::full({1, 2, 3}, 1.0f);
    bassinet::Tensor b = bassinet::Tensor::full({4, 1, 3, 5}, 2.0f);
    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.intl->shape(), std::vector<size_t>({4, 1, 2, 5}));
}
