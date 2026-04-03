#pragma once

#include "tensor.hpp"

namespace bassinet {
    void sgdStep(std::vector<Tensor*>& params, float lr);
}