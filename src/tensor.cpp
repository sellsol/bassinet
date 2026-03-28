#include "tensor.hpp"

const size_t MAX_TENSOR_PRINT_SIZE = 1000;

Tensor::Tensor(const std::vector<size_t>& shape, float defaultVal) : _shape{shape} {
    if (shape.size() == 0) throw std::invalid_argument("Tensor: Empty shape given");

    size_t size = 1;
    _stride = std::vector<size_t>(shape.size());
    for (size_t i = _stride.size(); i-- > 0; ) {
        if (i != _stride.size() - 1) _stride[i] = _stride[i + 1];
        _stride[i] = size;
        size *= _shape[i];
    }

    _data = std::vector<float>(size, defaultVal);
}


const std::vector<size_t>& Tensor::shape() {
    return _shape;
}

const std::vector<size_t>& Tensor::stride() {
    return _stride;
}

size_t Tensor::size() {
    return _data.size();
}


float& Tensor::at(const std::vector<size_t>& loc) {
    if (loc.size() != _shape.size()) throw std::invalid_argument("Tensor::at: Index does not match tensor shape");

    size_t trueIdx{0};
    for (size_t i = 0; i < loc.size(); ++i) {
        trueIdx += loc[i] * _stride[i];
    }
    return _data[trueIdx];
}

float Tensor::at(const std::vector<size_t>& loc) const {
    if (loc.size() != _shape.size()) throw std::invalid_argument("Tensor::at: Index does not match tensor shape");

    size_t trueIdx{0};
    for (size_t i = 0; i < loc.size(); ++i) {
        trueIdx += loc[i] * _stride[i];
    }
    return _data[trueIdx];
}


void Tensor::reshape(const std::vector<size_t>& newShape) {
    size_t newSize{1};
    std::vector<size_t> newStride{std::vector<size_t>(newShape.size())};
    for (size_t i = newStride.size(); i-- > 0; ) {
        if (i != newStride.size() - 1) newStride[i] = newStride[i + 1];
        newStride[i] = newSize;
        newSize *= _shape[i];
    }

    if (newSize != _data.size()) throw std::invalid_argument("Tensor::reshape:: New shape does not have equal number of elements");
    _shape = newShape;
    _stride = newStride;
}

Tensor Tensor::operator+(const Tensor& other) {
    std::vector<size_t> resShape(std::max(_shape.size(), other._shape.size()));

    std::vector<size_t> thisBroadcastStride(resShape.size());
    std::vector<size_t> otherBroadcastStride(resShape.size());
    size_t thisOffset{1};
    size_t otherOffset{1};
    for (size_t i = 0; i < resShape.size(); ++i) {
        size_t thisDimShape, otherDimShape;
        if (_shape.size() >= i + 1) { // _shape.size() - 1 - i >= 0
            thisDimShape = _shape[_shape.size() - 1 - i];
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = (thisDimShape == 1) ? 0 : thisOffset;
            if (thisDimShape != 1) thisOffset *= thisDimShape;
        } else {
            thisDimShape = 1; // actually 0, but easier comparison like this
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = 0;
        }

        if (other._shape.size() >= i + 1) {
            otherDimShape = other._shape[other._shape.size() - 1 - i];
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = (otherDimShape == 1) ? 0 : otherOffset;
            if (otherDimShape != 1) otherOffset *= otherDimShape;
        } else {
            otherDimShape = 1;
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = 0;
        }

        if (thisDimShape != otherDimShape && thisDimShape != 1 && otherDimShape != 1) throw std::invalid_argument("Tensor::matmul: Tensor batch dimensions not the same or broadcastable");

        resShape[resShape.size() - 1 - i] = std::max(thisDimShape, otherDimShape);
    }

    Tensor res(resShape);
    for (size_t idx = 0; idx < res._data.size(); ++idx) {
        size_t remainder{idx};
        size_t thisIdx{0}, otherIdx{0};
        for (size_t i = resShape.size(); i-- > 0; ) {
            size_t coord = remainder % resShape[i];
            thisIdx += coord * thisBroadcastStride[i];
            otherIdx += coord * otherBroadcastStride[i];
            remainder /= resShape[i];
        }
        res._data[idx] = _data[thisIdx] + other._data[otherIdx];
    }
    return res;
}

Tensor Tensor::matmul(const Tensor& other) {
    size_t N, K, M; // this shape (M, K), other shape (K, N), result shape (M, N)
    bool thisPromoted{false}, otherPromoted{false};

    if (_shape.size() == 1) {
        thisPromoted = true;
        M = 1;
        K = _shape[0];
    } else {
        M = _shape[_shape.size() - 2];
        K = _shape[_shape.size() - 1];
    }

    if (other._shape.size() == 1) {
        if (K != other._shape[0]) throw std::invalid_argument("Tensor::matmul: Tensor dimensions not overlapping");
        otherPromoted = true;
        N = 1;
    } else {
        if (K != other._shape[other._shape.size() - 2]) throw std::invalid_argument("Tensor::matmul: Tensor dimensions not overlapping");
        N = other._shape[other._shape.size() - 1];
    }

    std::vector<size_t> resShape(std::max(_shape.size(), other._shape.size()));
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
        if (_shape.size() >= i + 1) { // _shape.size() - 1 - i >= 0
            thisDimShape = _shape[_shape.size() - 1 - i];
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = thisOffset;
            thisOffset *= _shape[_shape.size() - 1 - i];
        } else {
            thisDimShape = 1; // actually 0, but easier comparison like this
            thisBroadcastStride[thisBroadcastStride.size() - 1 - i] = 0;
        }

        if (other._shape.size() >= i + 1) {
            otherDimShape = other._shape[other._shape.size() - 1 - i];
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = otherOffset;
            otherOffset *= other._shape[other._shape.size() - 1 - i];
        } else {
            otherDimShape = 1;
            otherBroadcastStride[otherBroadcastStride.size() - 1 - i] = 0;
        }

        if (thisDimShape != otherDimShape && thisDimShape != 1 && otherDimShape != 1) throw std::invalid_argument("Tensor::matmul: Tensor batch dimensions not the same or broadcastable");

        resShape[resShape.size() - 1 - i] = std::max(thisDimShape, otherDimShape);
        batchCount *= resShape[resShape.size() - 1 - i];
    }

    Tensor res(resShape);

    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t batchRemainder{batch};
        size_t resBatchOffset{0}, thisBatchOffset{0}, otherBatchOffset{0};
        if (resShape.size() > 2) {
            for (size_t i = 0; i < resShape.size() - 2; ++i) {
                resBatchOffset += (batchRemainder % resShape[i]) * res._stride[i];
                thisBatchOffset += (batchRemainder % resShape[i]) * thisBroadcastStride[i];
                otherBatchOffset += (batchRemainder % resShape[i]) * otherBroadcastStride[i];
                batchRemainder /= resShape[i];
            }
        }

        for (size_t resRow = 0; resRow < M; ++resRow) {
            for (size_t resCol = 0; resCol < N; ++resCol) {
                for (size_t k = 0; k < K; ++k) {
                    res._data[resBatchOffset + (resRow * (res._stride.size() > 1 ? res._stride[res._stride.size() - 2] : 0) + resCol)]
                    += _data[thisBatchOffset + (resRow * K + k)]
                    * other._data[otherBatchOffset + (k * N) + resCol];
                }
            }
        }
    }

    if (otherPromoted && thisPromoted) { res._shape = {1}; }
    else if (otherPromoted) { res._shape.pop_back(); res._stride.pop_back(); }
    else if (thisPromoted) { res._shape.erase(res._shape.begin()); res._stride.erase(res._stride.begin()); }

    return res;
}


void printTensor(std::ostream& out, const std::vector<float>& data,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& stride,
    std::size_t dim, std::size_t offset
) {
    out << std::string(dim * 2, ' ') << "[";
    if (dim == shape.size() - 1) {
        for (std::size_t i = 0; i < shape[dim]; ++i) {
            if (i > 0) out << ", ";
            out << data[offset + i * stride[dim]];
        }
    } else {
        out << "\n";
        for (std::size_t i = 0; i < shape[dim]; ++i) {
            if (i > 0) out << ",\n";
            printTensor(out, data, shape, stride, dim + 1, offset + i * stride[dim]);
        }
        out << "\n" << std::string(dim * 2, ' ');
    }
    out << "]";
}
std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    if (t._data.size() > MAX_TENSOR_PRINT_SIZE) {
        out << "[Tensor too large to print - size > 1000]";
        return out;
    }

    printTensor(out, t._data, t._shape, t._stride, 0, 0);
    return out;
}
