#include <gtest/gtest.h>
#include "tensor.hpp"

TEST(TensorTest, Construct) {
    bassinet::Tensor t({2, 3}, 5.0f);
    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(t.size(), 6);
}
TEST(TensorTest, ConstructDefault) {
    bassinet::Tensor t({2, 2}, 7.5f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 7.5f);
        }
    }
}


TEST(TensorTest, At) {
    bassinet::Tensor t({2, 2}, 4.2f);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 4.2f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.2f);
}
TEST(TensorTest, AtModify) {
    bassinet::Tensor t({2, 2});
    t.at({0, 1}) = 3.14f;
    EXPECT_FLOAT_EQ(t.at({0, 1}), 3.14f);
}


TEST(TensorTest, Addition) {
    bassinet::Tensor a({2, 2}, 1.0f);
    bassinet::Tensor b({2, 2}, 2.0f);
    bassinet::Tensor c = a + b;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(c.at({i, j}), 3.0f);
        }
    }
}
TEST(TensorTest, AdditionBroadcastScalar) {
    bassinet::Tensor a({2, 3}, 1.0f);
    bassinet::Tensor b({1}, 2.0f);
    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(c.at({i, j}), 3.0f);
}
TEST(TensorTest, AdditionBroadcastCol) {
    bassinet::Tensor a({2, 3}, 1.0f);
    bassinet::Tensor b({2, 1}, 2.0f);
    a.at({1, 2}) = 4.0f;

    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_FLOAT_EQ(c.at({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(c.at({1, 2}), 6.0f);
}
TEST(TensorTest, AdditionBroadcastBoth) {
    bassinet::Tensor a({2, 1, 3}, 1.0f);
    bassinet::Tensor b({1, 4, 1}, 2.0f);
    bassinet::Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 4, 3}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 3; ++k)
                EXPECT_FLOAT_EQ(c.at({i, j, k}), 3.0f);
}
TEST(TensorTest, AdditionShapeMismatch) {
    bassinet::Tensor a({2, 2}, 1.0f);
    bassinet::Tensor b({2, 3}, 2.0f);
    EXPECT_THROW(a + b, std::invalid_argument);
}


TEST(TensorTest, Matmul1Dx1D) {
    bassinet::Tensor a({3}, 0.0f);
    a.at({0}) = 1.0f; a.at({1}) = 2.0f; a.at({2}) = 3.0f;

    bassinet::Tensor b({3}, 0.0f);
    b.at({0}) = 4.0f; b.at({1}) = 5.0f; b.at({2}) = 6.0f;

    bassinet::Tensor result = a.matmul(b);
    // EXPECT_EQ(result.shape(), std::vector<size_t>({1}));
    EXPECT_FLOAT_EQ(result.at({0}), 32);
}
TEST(TensorTest, Matmul2Dx2D) {
    bassinet::Tensor a({2, 3}, 0.0f);
    a.at({0,0}) = 1; a.at({0,1}) = 2; a.at({0,2}) = 3;
    a.at({1,0}) = 4; a.at({1,1}) = 5; a.at({1,2}) = 6;

    bassinet::Tensor b({3, 2}, 0.0f);
    b.at({0,0}) = 7;  b.at({0,1}) = 8;
    b.at({1,0}) = 9;  b.at({1,1}) = 10;
    b.at({2,0}) = 11; b.at({2,1}) = 12;

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2,2}));
    EXPECT_FLOAT_EQ(result.at({0,0}), 58);
    EXPECT_FLOAT_EQ(result.at({0,1}), 64);
    EXPECT_FLOAT_EQ(result.at({1,0}), 139);
    EXPECT_FLOAT_EQ(result.at({1,1}), 154);
}
TEST(TensorTest, Matmul1Dx2D) {
    bassinet::Tensor a({3}, 0.0f);
    a.at({0}) = 1; a.at({1}) = 2; a.at({2}) = 3;

    bassinet::Tensor b({3, 2}, 0.0f);
    b.at({0,0}) = 4; b.at({0,1}) = 5;
    b.at({1,0}) = 6; b.at({1,1}) = 7;
    b.at({2,0}) = 8; b.at({2,1}) = 9;

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2}));
    EXPECT_FLOAT_EQ(result.at({0}), 40);
    EXPECT_FLOAT_EQ(result.at({1}), 46);
}
TEST(TensorTest, Matmul2Dx1D) {
    bassinet::Tensor a({2, 3}, 0.0f);
    a.at({0,0}) = 1; a.at({0,1}) = 2; a.at({0,2}) = 3;
    a.at({1,0}) = 4; a.at({1,1}) = 5; a.at({1,2}) = 6;

    bassinet::Tensor b({3}, 0.0f);
    b.at({0}) = 7; b.at({1}) = 8; b.at({2}) = 9;

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2}));
    EXPECT_FLOAT_EQ(result.at({0}), 50);
    EXPECT_FLOAT_EQ(result.at({1}), 122);
}
TEST(TensorTest, MatmulBatched3Dx3D) {
    bassinet::Tensor a({2, 2, 2}, 0.0f);
    // batch 0
    a.at({0,0,0}) = 0; a.at({0,0,1}) = 2;
    a.at({0,1,0}) = -2; a.at({0,1,1}) = -5;
    // batch 1
    a.at({1,0,0}) = -5; a.at({1,0,1}) = -5;
    a.at({1,1,0}) = -1; a.at({1,1,1}) = 2;

    bassinet::Tensor b({2, 2, 2}, 0.0f);
    // batch 0
    b.at({0,0,0}) = 6; b.at({0,0,1}) = -6;
    b.at({0,1,0}) = 3; b.at({0,1,1}) = 0;
    // batch 1
    b.at({1,0,0}) = -2; b.at({1,0,1}) = -3;
    b.at({1,1,0}) = 3; b.at({1,1,1}) = 5;

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2,2,2}));
    EXPECT_FLOAT_EQ(result.at({0,0,0}), 6);
    EXPECT_FLOAT_EQ(result.at({1,1,1}), 13);
}
TEST(TensorTest, MatmulShapeMismatch) {
    bassinet::Tensor a({2, 3}, 1.0f);
    bassinet::Tensor b({4, 2}, 1.0f);
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}
TEST(TensorTest, MatmulBroadcastBatchLeft) {
    bassinet::Tensor a({2, 3, 4}, 1.0f);
    bassinet::Tensor b({4, 5}, 2.0f);

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2, 3, 5}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 5; ++k)
                EXPECT_FLOAT_EQ(result.at({i, j, k}), 8.0f);
}

TEST(TensorTest, MatmulBroadcastBatchRight) {
    bassinet::Tensor a({4, 3}, 1.0f);
    bassinet::Tensor b({2, 3, 5}, 2.0f);

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2, 4, 5}));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 5; ++k)
                EXPECT_FLOAT_EQ(result.at({i, j, k}), 6.0f);
}

TEST(TensorTest, MatmulBroadcastBothSides) {
    bassinet::Tensor a({1, 2, 3}, 1.0f);
    bassinet::Tensor b({4, 1, 3, 5}, 2.0f);

    bassinet::Tensor result = a.matmul(b);
    EXPECT_EQ(result.shape(), std::vector<size_t>({4, 1, 2, 5}));
}
