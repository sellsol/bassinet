#pragma once

#include "tensor.hpp"

namespace bassinet {
    // enum Reduction { MEAN, SUM };

    Tensor mseLoss(Tensor& pred, Tensor& target);
}