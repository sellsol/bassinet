#include "tensor.hpp"

const size_t MAX_TENSOR_PRINT_SIZE = 1000;

void flattenInitlistRec(const RecursiveList& rl, std::vector<float>& flattened, std::vector<size_t>& shape, size_t depth) {
    if (rl.vec.empty()) {
        flattened.push_back(rl.val);
        return;
    }

    if (depth == shape.size()) shape.push_back(rl.vec.size());
    else if (shape[depth] != rl.vec.size()) throw std::invalid_argument("Tensor: n-dim vector is not rectangular");

    for (const auto& child : rl.vec) {
        flattenInitlistRec(child, flattened, shape, depth + 1);
    }
}
bassinet::TensorIntl::TensorIntl(std::initializer_list<RecursiveList> data): _gradRequired{false} {
    std::vector<float> flattened;
    flattenInitlistRec(data, flattened, _shape, 0);
    _data = std::make_shared<std::vector<float>>(std::move(flattened));

    _stride = std::vector<size_t>(_shape.size());
    size_t size{1};
    for (size_t i = _stride.size(); i-- > 0; ) {
        _stride[i] = size;
        size *= _shape[i];
    }
}

bassinet::TensorIntl bassinet::TensorIntl::full(const std::vector<size_t>& shape, float val, bool gradRequired) {
    if (shape.size() == 0) throw std::invalid_argument("Tensor: Empty shape given");

    std::vector<size_t> stride(shape.size());
    size_t size = 1;
    for (size_t i = stride.size(); i-- > 0; ) {
        if (i != stride.size() - 1) stride[i] = stride[i + 1];
        stride[i] = size;
        size *= shape[i];
    }

    TensorIntl newTI;
    newTI._data = std::make_shared<std::vector<float>>(size, val);
    newTI._shape = shape;
    newTI._stride = stride;
    newTI._gradRequired = gradRequired;
    if (gradRequired) newTI._grad = std::vector<float>(size, 0.0f);
    return newTI;
}

bassinet::TensorIntl bassinet::TensorIntl::zeros(const std::vector<size_t>& shape, bool gradRequired) {
    return bassinet::TensorIntl::full(shape, 0, gradRequired);
}

bassinet::TensorIntl bassinet::TensorIntl::fromMove(const std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn, const std::vector<std::shared_ptr<TensorIntl>>& parents) {
    if (shape.size() != stride.size()) throw std::invalid_argument("Tensor: shape and stride must have the same number of dimensions");

    size_t maxIdx{0};
    for (size_t i = 0; i < shape.size(); ++i) {
        maxIdx += (shape[i] - 1) * stride[i];
    }
    if (maxIdx + 1 != data.size()) throw std::invalid_argument("Tensor: shape and stride does not match data size");

    TensorIntl newTI;
    newTI._data = std::make_shared<std::vector<float>>(data);
    newTI._shape = shape;
    newTI._stride = stride;
    newTI._gradRequired = gradRequired;
    if (gradRequired) newTI._grad = std::vector<float>(data.size(), 0.0f);
    newTI._gradFn = gradFn;
    newTI._parents = parents;
    return newTI;
}

bassinet::TensorIntl bassinet::TensorIntl::fromMove(const std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn, const std::vector<std::shared_ptr<TensorIntl>>& parents) {
    if (shape.size() != stride.size()) throw std::invalid_argument("Tensor: shape and stride must have the same number of dimensions");

    size_t maxIdx{0};
    for (size_t i = 0; i < shape.size(); ++i) {
        maxIdx += (shape[i] - 1) * stride[i];
    }
    if (maxIdx + 1 != data->size()) throw std::invalid_argument("Tensor: shape and stride does not match data size");

    TensorIntl newTI;
    newTI._data = data;
    newTI._shape = shape;
    newTI._stride = stride;
    newTI._gradRequired = gradRequired;
    if (gradRequired) newTI._grad = std::vector<float>(data->size(), 0.0f);
    newTI._gradFn = gradFn;
    newTI._parents = parents;
    return newTI;
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

    std::shared_ptr<bassinet::TensorIntl> transposed = std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl::fromMove(_data, _shape, _stride));
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


bassinet::Tensor::Tensor(std::initializer_list<RecursiveList> data) : intl{std::make_shared<bassinet::TensorIntl>(data)} {}

bassinet::Tensor::Tensor(std::shared_ptr<bassinet::TensorIntl> internal) : intl{internal} {}

bassinet::Tensor bassinet::Tensor::full(const std::vector<size_t>& shape, float val, bool gradRequired) {
    return Tensor(std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl::full(shape, val, gradRequired)));
}

bassinet::Tensor bassinet::Tensor::zeros(const std::vector<size_t>& shape, bool gradRequired) {
    return Tensor(std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl::zeros(shape, gradRequired)));
}

bassinet::Tensor bassinet::Tensor::fromMove(const std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn, const std::vector<std::shared_ptr<TensorIntl>>& parents) {
    return Tensor(std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl::fromMove(data, shape, stride, gradRequired, gradFn, parents)));
}

bassinet::Tensor bassinet::Tensor::fromMove(const std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired, std::function<void(std::vector<std::shared_ptr<bassinet::TensorIntl>>&, bassinet::TensorIntl&)> gradFn, const std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents) {
    return Tensor(std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl::fromMove(data, shape, stride, gradRequired, gradFn, parents)));
}


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

    return bassinet::Tensor::fromMove(
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

    std::shared_ptr<bassinet::TensorIntl> a = parents[0];
    std::shared_ptr<bassinet::TensorIntl> b = parents[1];
    bool aWas1D{a->shape().size() == 1}; // will check and unpromote at end
    bool bWas1D{b->shape().size() == 1};
    if (aWas1D) a = bassinet::Tensor::fromMove(a->data(), {1, a->shape()[0]}, {a->shape()[0], 1}).intl;
    if (bWas1D) b = bassinet::Tensor::fromMove(b->data(), {b->shape()[0], 1}, {1, 1}).intl;
    std::shared_ptr<bassinet::TensorIntl> bT{b->transpose(b->shape().size() - 1, b->shape().size() - 2)};

    // a shape = (M, K), b shape = (K, N), matmul(a,b) shape = (M, N)
    const size_t M{a->shape()[a->shape().size() - 2]};
    const size_t K{a->shape()[a->shape().size() - 1]};
    if (K != bT->shape()[bT->shape().size() - 1]) throw std::invalid_argument("Tensor::matmul: Tensor dimensions not overlapping");
    const size_t N{bT->shape()[b->shape().size() - 2]};

    std::vector<size_t> resShape(std::max(a->shape().size(), b->shape().size()), 1);
    resShape[resShape.size() - 2] = M;
    resShape[resShape.size() - 1] = N;

    std::vector<size_t> resStride(resShape.size());
    size_t resSize{1};
    for (size_t i = resShape.size(); i-- > 0; ) {
        resStride[i] = resSize;
        resSize *= resShape[i];
    }

    std::vector<size_t> aBroadcastStride(resShape.size(), 0);
    std::vector<size_t> bBroadcastStride(resShape.size(), 0);
    size_t aOffset{M * K};
    size_t bOffset{N * K}; // bT has shape (..., N, K)
    size_t batchCount{1};
    for (size_t i = 2; i < resShape.size(); ++i) {
        const size_t resDim = resShape.size() - 1 - i;

        size_t aDimShape{1}, bDimShape{1};
        if (a->shape().size() >= i + 1) {
            aDimShape = a->shape()[a->shape().size() - 1 - i];
            aBroadcastStride[resDim] = (aDimShape == 1) ? 0 : aOffset;
            if (aDimShape != 1) aOffset *= aDimShape;
        }
        if (bT->shape().size() >= i + 1) {
            bDimShape = bT->shape()[bT->shape().size() - 1 - i];
            bBroadcastStride[resDim] = (bDimShape == 1) ? 0 : bOffset;
            if (bDimShape != 1) bOffset *= bDimShape;
        }
        if (aDimShape != bDimShape && aDimShape != 1 && bDimShape != 1) throw std::invalid_argument("Tensor::matmul: Tensor batch dimensions not the same or broadcastable");

        resShape[resDim] = std::max(aDimShape, bDimShape);
        batchCount *= resShape[resDim];
    }

    resSize = 1; // recompute result layout since final resShape may be different
    for (size_t i = resShape.size(); i-- > 0; ) {
        resStride[i] = resSize;
        resSize *= resShape[i];
    }
    std::shared_ptr<std::vector<float>> resData{std::make_shared<std::vector<float>>(resSize, 0.0f)};

    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t batchRemainder{batch};
        size_t resBatchOffset{0}, aBatchOffset{0}, bBatchOffset{0};
        if (resShape.size() > 2) {
            for (size_t i = 0; i < resShape.size() - 2; ++i) {
                resBatchOffset += (batchRemainder % resShape[i]) * resStride[i];
                aBatchOffset += (batchRemainder % resShape[i]) * aBroadcastStride[i];
                bBatchOffset += (batchRemainder % resShape[i]) * bBroadcastStride[i];
                batchRemainder /= resShape[i];
            }
        }

        for (size_t resRow = 0; resRow < M; ++resRow) {
            for (size_t resCol = 0; resCol < N; ++resCol) {
                size_t resIdx{resBatchOffset
                    + resRow * resStride[resStride.size() - 2]
                    + resCol * resStride[resStride.size() - 1]};

                for (size_t k = 0; k < K; ++k) {
                    const size_t aIdx = aBatchOffset
                        + resRow * a->stride()[a->shape().size() - 2]
                        + k * a->stride()[a->shape().size() - 1];
                    const size_t bIdx = bBatchOffset
                        + resCol * bT->stride()[bT->shape().size() - 2]
                        + k * bT->stride()[bT->shape().size() - 1];
                    (*resData)[resIdx] += (*a->data())[aIdx] * (*bT->data())[bIdx];
                }
            }
        }
    }

    if (aWas1D && bWas1D) {
        resShape = {1}; resStride = {1};
    } else if (aWas1D) {
        resShape.erase(resShape.end() - 2);
        resStride.erase(resStride.end() - 2);
    } else if (bWas1D) {
        resShape.pop_back(); resStride.pop_back();
    }

    return bassinet::Tensor::fromMove(
        resData, resShape, resStride,
        parents[0]->gradRequired() || parents[1]->gradRequired(),
        matmulBackward, parents
    );
}

void matmulBackward(std::vector<std::shared_ptr<bassinet::TensorIntl>>& parents, bassinet::TensorIntl& child) {
    if (parents.size() != 2) throw std::invalid_argument("matmulBackward: Operation only supports two parents");

    std::shared_ptr<bassinet::TensorIntl> a = parents[0];
    std::shared_ptr<bassinet::TensorIntl> b = parents[1];
    bool aWas1D{a->shape().size() == 1}; // see matmulForward for similar promotion and depromotion logic
    bool bWas1D{b->shape().size() == 1};
    if (aWas1D) a = bassinet::Tensor::fromMove(a->data(), {1, a->shape()[0]}, {a->shape()[0], 1}).intl;
    if (bWas1D) b = bassinet::Tensor::fromMove(b->data(), {b->shape()[0], 1}, {1, 1}).intl;

    bassinet::Tensor gradTensor;
    if (child.shape().size() == 1) {
        if (aWas1D && bWas1D) { // see end of matmulBackward for child depromotion logic
            gradTensor = bassinet::Tensor::fromMove(child.grad(), {1, 1}, {1, 1});
        } else if (aWas1D) { // [1, N] before depromotion
            gradTensor = bassinet::Tensor::fromMove(child.grad(), {1, child.shape()[0]}, {child.shape()[0], 1});
        } else if (bWas1D) { // [N, 1] before depromotion
            gradTensor = bassinet::Tensor::fromMove(child.grad(), {child.shape()[0], 1}, {1, 1});
        }
    }

    if (parents[0]->gradRequired()) {
        bassinet::Tensor bT{b->transpose(b->shape().size() - 1, b->shape().size() - 2)};
        bassinet::Tensor res(matmulForward({gradTensor.intl, bT.intl}));

        if (aWas1D) {
            std::vector<size_t> resShape{res.intl->shape()};
            std::vector<size_t> resStride{res.intl->stride()};
            resShape.erase(resShape.end() - 2);
            resStride.erase(resStride.end() - 2);
            res = bassinet::Tensor::fromMove(res.intl->data(), resShape, resStride);
        }
        parents[0]->addToGrad(*res.intl->data());
    }
    if (parents[1]->gradRequired()) {
        bassinet::Tensor aT{a->transpose(a->shape().size() - 1, a->shape().size() - 2)};
        bassinet::Tensor res(matmulForward({aT.intl, gradTensor.intl}));

        if (bWas1D) {
            std::vector<size_t> resShape{res.intl->shape()};
            std::vector<size_t> resStride{res.intl->stride()};
            resShape.pop_back();
            resStride.pop_back();
            res = bassinet::Tensor::fromMove(res.intl->data(), resShape, resStride);
        }
        parents[1]->addToGrad(*res.intl->data());
    }
};

bassinet::Tensor bassinet::Tensor::matmul(bassinet::Tensor& other) {
    return matmulForward({(*this).intl, other.intl});
}
