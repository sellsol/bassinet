#include "tensor.hpp"

const size_t MAX_TENSOR_PRINT_SIZE = 1000;

bassinet::TensorIntl::TensorIntl(const std::vector<size_t>& shape, float defaultVal, bool gradRequired) : _shape{shape}, _gradRequired{gradRequired} {
    if (shape.size() == 0) throw std::invalid_argument("Tensor: Empty shape given");

    size_t size = 1;
    _stride = std::vector<size_t>(shape.size());
    for (size_t i = _stride.size(); i-- > 0; ) {
        if (i != _stride.size() - 1) _stride[i] = _stride[i + 1];
        _stride[i] = size;
        size *= _shape[i];
    }
    _data = std::make_shared<std::vector<float>>(size, defaultVal);

    if (_gradRequired) _grad = std::vector<float>(size, 0.0f);
}

bassinet::TensorIntl::TensorIntl(std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn, const std::vector<std::shared_ptr<TensorIntl>>& parents) : _data{std::make_shared<std::vector<float>>(data)}, _shape{shape}, _stride{stride}, _gradRequired{gradRequired}, _gradFn{gradFn}, _parents{parents} {
    if (shape.size() != stride.size()) throw std::invalid_argument("Tensor: shape and stride must have the same number of dimensions");

    size_t maxIdx{0};
    for (size_t i = 0; i < shape.size(); ++i) {
        maxIdx += (shape[i] - 1) * stride[i];
    }
    if (maxIdx + 1 != data.size()) throw std::invalid_argument("Tensor: shape and stride does not match data size");

    if (_gradRequired) _grad = std::vector<float>(size(), 0.0f);
}

bassinet::TensorIntl::TensorIntl(std::shared_ptr<std::vector<float>> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<bassinet::TensorIntl>>&, bassinet::TensorIntl&)> gradFn, const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) : _data{data}, _shape{shape}, _stride{stride}, _gradRequired{gradRequired}, _gradFn{gradFn}, _parents{parents} {
    if (shape.size() != stride.size()) throw std::invalid_argument("Tensor: shape and stride must have the same number of dimensions");

    size_t maxIdx{0};
    for (size_t i = 0; i < shape.size(); ++i) {
        maxIdx += (shape[i] - 1) * stride[i];
    }
    if (maxIdx + 1 != data->size()) throw std::invalid_argument("Tensor: shape and stride does not match data size");

    if (_gradRequired) _grad = std::vector<float>(size(), 0.0f);
}


const std::shared_ptr<std::vector<float>>& bassinet::TensorIntl::data() const {
    return _data;
}

const std::vector<size_t>& bassinet::TensorIntl::shape() const {
    return _shape;
}

const std::vector<size_t>& bassinet::TensorIntl::stride() const {
    return _stride;
}

size_t bassinet::TensorIntl::size() const {
    return _data->size();
}

bool bassinet::TensorIntl::gradRequired() const {
    return _gradRequired;
}

const std::vector<float>& bassinet::TensorIntl::grad() const {
    return _grad;
}


float& bassinet::TensorIntl::at(const std::vector<size_t>& loc) {
    if (loc.size() != _shape.size()) throw std::invalid_argument("Tensor::at: Index does not match tensor shape");

    size_t trueIdx{0};
    for (size_t i = 0; i < loc.size(); ++i) {
        trueIdx += loc[i] * _stride[i];
    }
    return (*_data)[trueIdx];
}

float bassinet::TensorIntl::at(const std::vector<size_t>& loc) const {
    if (loc.size() != _shape.size()) throw std::invalid_argument("Tensor::at: Index does not match tensor shape");

    size_t trueIdx{0};
    for (size_t i = 0; i < loc.size(); ++i) {
        trueIdx += loc[i] * _stride[i];
    }
    return (*_data)[trueIdx];
}

std::shared_ptr<bassinet::TensorIntl> bassinet::TensorIntl::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= _shape.size() || dim1 >= _shape.size()) throw std::out_of_range("Tensor::transpose: Dimensions to transpose out of bounds");

    std::shared_ptr<bassinet::TensorIntl> transposed = std::make_shared<bassinet::TensorIntl>(_data, _shape, _stride);
    std::swap(transposed->_shape[dim0], transposed->_shape[dim1]);
    std::swap(transposed->_stride[dim0], transposed->_stride[dim1]);

    return transposed;
}


void bassinet::TensorIntl::addToData(const std::vector<float>& additions) {
    if (size() != additions.size()) throw std::invalid_argument("Tensor::addToData: Data to add does not match existing data size");

    for (size_t i = 0; i < size(); ++i) {
        (*_data)[i] += additions[i];
    }
}

void bassinet::TensorIntl::zeroGrad() {
    std::fill(_grad.begin(), _grad.end(), 0.0f);
}

void bassinet::TensorIntl::addToGrad(const std::vector<float>& additions) {
    if (additions.size() != _grad.size()) throw std::invalid_argument("Tensor::addToGrad: Gradients to add does not match existing gradient shape");

    for (size_t i = 0; i < _grad.size(); ++i) {
        _grad[i] += additions[i];
    }
}

void bassinet::TensorIntl::backward() {
    if (!_gradRequired || !_gradFn) return; // TODO: bugfix why _gradRequired might be true but not _gradFn

    // If this is a scalar and _grad is all zeros, set to 1.0 to seed the backward pass
    // TODO: figure out why this is necessary
    if (_grad.size() == 1 && _grad[0] == 0.0f) {
        _grad[0] = 1.0f;
    }

    _gradFn(_parents, *this);
    for (std::shared_ptr<TensorIntl>& parent : _parents) {
        parent->backward();
    }
}


void printTensorIntl(std::ostream& out, const std::vector<float>& data,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& stride,
    size_t dim, size_t offset
) {
    out << std::string(dim * 2, ' ') << "[";
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i > 0) out << ", ";
            out << data[offset + i * stride[dim]];
        }
    } else {
        out << "\n";
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i > 0) out << ",\n";
            printTensorIntl(out, data, shape, stride, dim + 1, offset + i * stride[dim]);
        }
        out << "\n" << std::string(dim * 2, ' ');
    }
    out << "]";
}

std::ostream& bassinet::operator<<(std::ostream& out, const bassinet::TensorIntl& t) {
    if (t.size() > MAX_TENSOR_PRINT_SIZE) {
        out << "[Tensor too large to print - size > 1000]";
        return out;
    }

    printTensorIntl(out, *t.data(), t.shape(), t.stride(), 0, 0);
    return out;
}


bassinet::Tensor::Tensor(const std::vector<size_t>& shape, float defaultVal, bool gradRequired) : intl{std::make_shared<bassinet::TensorIntl>(shape, defaultVal, gradRequired)} {}

bassinet::Tensor::Tensor(std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn, const std::vector<std::shared_ptr<TensorIntl>>& parents) : intl{std::make_shared<bassinet::TensorIntl>(data, shape, stride, gradRequired, gradFn, parents)} {}

bassinet::Tensor::Tensor(std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<bassinet::TensorIntl>>&, bassinet::TensorIntl&)> gradFn, const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) : intl{std::make_shared<bassinet::TensorIntl>(data, shape, stride, gradRequired, gradFn, parents)} {}

bassinet::Tensor::Tensor(std::shared_ptr<bassinet::TensorIntl> internal) : intl{internal} {}


void addBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child);
bassinet::Tensor addForward(const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) {
    if (parents.size() != 2) throw std::invalid_argument("addForward: Operation only supports two parents");

    std::vector<size_t> resShape(std::max(parents[0]->shape().size(), parents[1]->shape().size()));

    std::vector<size_t> thisBroadcastStride(resShape.size());
    std::vector<size_t> otherBroadcastStride(resShape.size());
    size_t thisOffset{1};
    size_t otherOffset{1};
    for (size_t i = 0; i < resShape.size(); ++i) {
        size_t thisDimShape, otherDimShape;
        if (parents[0]->shape().size() >= i + 1) { // shape.size() - 1 - i >= 0
            thisDimShape = parents[0]->shape()[parents[0]->shape().size() - 1 - i];
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = (thisDimShape == 1) ? 0 : thisOffset;
            if (thisDimShape != 1) thisOffset *= thisDimShape;
        } else {
            thisDimShape = 1; // actually 0, but easier comparison like this
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = 0;
        }

        if (parents[1]->shape().size() >= i + 1) {
            otherDimShape = parents[1]->shape()[parents[1]->shape().size() - 1 - i];
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = (otherDimShape == 1) ? 0 : otherOffset;
            if (otherDimShape != 1) otherOffset *= otherDimShape;
        } else {
            otherDimShape = 1;
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = 0;
        }

        if (thisDimShape != otherDimShape && thisDimShape != 1 && otherDimShape != 1) throw std::invalid_argument("Tensor::matmul: Tensor batch dimensions not the same or broadcastable");

        resShape[resShape.size() - 1 - i] = std::max(thisDimShape, otherDimShape);
    }

    size_t resSize{1};
    std::vector<size_t> resStride{std::vector<size_t>(resShape.size())};
    for (size_t i = resStride.size(); i-- > 0; ) {
        if (i != resStride.size() - 1) resStride[i] = resStride[i + 1];
        resStride[i] = resSize;
        resSize *= resShape[i];
    }
    std::shared_ptr<std::vector<float>> resData{std::make_shared<std::vector<float>>(resSize)};

    for (size_t idx = 0; idx < (*resData).size(); ++idx) {
        size_t remainder{idx};
        size_t thisIdx{0}, otherIdx{0};
        for (size_t i = resShape.size(); i-- > 0; ) {
            size_t coord = remainder % resShape[i];
            thisIdx += coord * thisBroadcastStride[i];
            otherIdx += coord * otherBroadcastStride[i];
            remainder /= resShape[i];
        }
        (*resData)[idx] = (*parents[0]->data())[thisIdx] + (*parents[1]->data())[otherIdx];
    }

    return bassinet::Tensor(
        resData, resShape, resStride,
        parents[0]->gradRequired() || parents[1]->gradRequired(),
        addBackward, parents
    );
}

void addBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child) {
    if (!parents.size()) throw std::invalid_argument("addBackward: Operation only supports two parents");

    if (parents[0]->gradRequired()) parents[0]->addToGrad(child.grad());
    if (parents[1]->gradRequired()) parents[1]->addToGrad(child.grad());
}

bassinet::Tensor bassinet::Tensor::operator+(bassinet::Tensor& other) {
   return addForward({(*this).intl, other.intl});
}


void matmulBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child);
bassinet::Tensor matmulForward(const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) {
    if (parents.size() != 2) throw std::invalid_argument("matmulBackward: Operation only supports two parents");

    size_t N, K, M; // this shape (M, K), other shape (K, N), result shape (M, N)
    bool thisPromoted{false}, otherPromoted{false};

    if (parents[0]->shape().size() == 1) {
        thisPromoted = true;
        M = 1;
        K = parents[0]->shape()[0];
    } else {
        M = parents[0]->shape()[parents[0]->shape().size() - 2];
        K = parents[0]->shape()[parents[0]->shape().size() - 1];
    }

    if (parents[1]->shape().size() == 1) {
        if (K != parents[1]->shape()[0]) throw std::invalid_argument("Tensor::matmul: Tensor dimensions not overlapping");
        otherPromoted = true;
        N = 1;
    } else {
        if (K != parents[1]->shape()[parents[1]->shape().size() - 2]) throw std::invalid_argument("Tensor::matmul: Tensor dimensions not overlapping");
        N = parents[1]->shape()[parents[1]->shape().size() - 1];
    }

    std::vector<size_t> resShape(std::max(parents[0]->shape().size(), parents[1]->shape().size()));
    if (resShape.size() == 1) {
        resShape[0] = 1;
    } else {
        resShape[resShape.size() - 2] = M;
        resShape[resShape.size() - 1] = N;
    }

    std::vector<size_t> thisBroadcastStride(resShape.size());
    std::vector<size_t> otherBroadcastStride(resShape.size());
    size_t thisOffset{M * K};
    size_t otherOffset{K * N};
    size_t batchCount{1};
    for (size_t i = 2; i < resShape.size(); ++i) {
        size_t thisDimShape, otherDimShape;
        if (parents[0]->shape().size() >= i + 1) { // _shape.size() - 1 - i >= 0
            thisDimShape = parents[0]->shape()[parents[0]->shape().size() - 1 - i];
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = thisOffset;
            thisOffset *= parents[0]->shape()[parents[0]->shape().size() - 1 - i];
        } else {
            thisDimShape = 1; // actually 0, but easier comparison like this
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = 0;
        }

        if (parents[1]->shape().size() >= i + 1) {
            otherDimShape = parents[1]->shape()[parents[1]->shape().size() - 1 - i];
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = otherOffset;
            otherOffset *= parents[1]->shape()[parents[1]->shape().size() - 1 - i];
        } else {
            otherDimShape = 1;
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = 0;
        }

        if (thisDimShape != otherDimShape && thisDimShape != 1 && otherDimShape != 1) throw std::invalid_argument("Tensor::matmul: Tensor batch dimensions not the same or broadcastable");

        resShape[resShape.size() - 1 - i] = std::max(thisDimShape, otherDimShape);
        batchCount *= resShape[resShape.size() - 1 - i];
    }

    size_t resSize{1};
    std::vector<size_t> resStride{std::vector<size_t>(resShape.size())};
    for (size_t i = resStride.size(); i-- > 0; ) {
        if (i != resStride.size() - 1) resStride[i] = resStride[i + 1];
        resStride[i] = resSize;
        resSize *= resShape[i];
    }
    std::shared_ptr<std::vector<float>> resData{std::make_shared<std::vector<float>>(resSize)};

    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t batchRemainder{batch};
        size_t resBatchOffset{0}, thisBatchOffset{0}, otherBatchOffset{0};
        if (resShape.size() > 2) {
            for (size_t i = 0; i < resShape.size() - 2; ++i) {
                resBatchOffset += (batchRemainder % resShape[i]) * resStride[i];
                thisBatchOffset += (batchRemainder % resShape[i]) * thisBroadcastStride[i];
                otherBatchOffset += (batchRemainder % resShape[i]) * otherBroadcastStride[i];
                batchRemainder /= resShape[i];
            }
        }

        for (size_t resRow = 0; resRow < M; ++resRow) {
            for (size_t resCol = 0; resCol < N; ++resCol) {
                for (size_t k = 0; k < K; ++k) {
                    (*resData)[resBatchOffset + (resRow * (resStride.size() > 1 ? resStride[resStride.size() - 2] : 0) + resCol)]
                    += (*parents[0]->data())[thisBatchOffset + (resRow * K + k)]
                    * (*parents[1]->data())[otherBatchOffset + (k * N) + resCol];
                }
            }
        }
    }

    if (otherPromoted && thisPromoted) { resShape = {1}; }
    else if (otherPromoted) { resShape.pop_back(); resStride.pop_back(); }
    else if (thisPromoted) { resShape.erase(resShape.begin()); resStride.erase(resStride.begin()); }

    return bassinet::Tensor(
        resData, resShape, resStride,
        parents[0]->gradRequired() || parents[1]->gradRequired(),
        matmulBackward, parents
    );
}

void matmulBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child) {
    if (parents.size() != 2) throw std::invalid_argument("matmulBackward: Operation only supports two parents");

    if (parents[0]->gradRequired()) {
        bassinet::Tensor gradTensor(child.grad(), child.shape(), child.stride());
        if (parents[1]->shape().size() >= 2) {
            bassinet::Tensor bT{parents[1]->transpose(parents[1]->shape().size() - 1, parents[1]->shape().size() - 2)};
            parents[0]->addToGrad(*matmulForward({gradTensor.intl, bT.intl}).intl->data());
        } else {
            bassinet::Tensor bT{parents[1]};
            parents[0]->addToGrad(*matmulForward({gradTensor.intl, bT.intl}).intl->data());
        }
    }
    if (parents[1]->gradRequired()) {
        bassinet::Tensor gradTensor(child.grad(), child.shape(), child.stride());
        if (parents[0]->shape().size() >= 2) {
            bassinet::Tensor aT{parents[0]->transpose(parents[0]->shape().size() - 1, parents[0]->shape().size() - 2)};
            parents[1]->addToGrad(*matmulForward({aT.intl, gradTensor.intl}).intl->data());
        } else {
            bassinet::Tensor aT{parents[0]};
            parents[1]->addToGrad(*matmulForward({aT.intl, gradTensor.intl}).intl->data());
        }
    }
};

bassinet::Tensor bassinet::Tensor::matmul(bassinet::Tensor& other) {
    return matmulForward({(*this).intl, other.intl});
}
