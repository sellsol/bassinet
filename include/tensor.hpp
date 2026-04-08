#pragma once

#include <iostream>
#include <vector>

extern const size_t MAX_TENSOR_PRINT_SIZE;

namespace bassinet {
    class TensorIntl {
    private:
        std::shared_ptr<std::vector<float>> _data; // row-order
        std::vector<size_t> _shape;
        std::vector<size_t> _stride;

        bool _gradRequired;
        std::vector<float> _grad; // if access by tensor index required, just use same calcs as for _data
        std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> _gradFn; // set by operator forward version
        std::vector<std::shared_ptr<TensorIntl>> _parents;

    public:
        TensorIntl() = default;
        template <typename T> // function defined in same file
        TensorIntl(const std::vector<T> data, bool gradRequired = false);
        static TensorIntl zeros(const std::vector<size_t>& shape, bool gradRequired = false);
        static TensorIntl fromMove(const std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {}); // TODO
        static TensorIntl fromMove(const std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});

        const std::shared_ptr<std::vector<float>>& data() const;
        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& stride() const;
        size_t size() const;
        bool gradRequired() const;
        const std::vector<float>& grad() const;

        float& at(const std::vector<size_t>& loc);
        float at(const std::vector<size_t>& loc) const;
        std::shared_ptr<TensorIntl> transpose(size_t dim0, size_t dim1) const;

        void addToData(const std::vector<float>& additions);
        void zeroGrad();
        void addToGrad(const std::vector<float>& additions);
        void backward();
    };

    std::ostream& operator<<(std::ostream& out, const TensorIntl& t); // namespace resolved through ADL

    class Tensor {
    public:
        std::shared_ptr<TensorIntl> intl;

        Tensor(std::shared_ptr<TensorIntl> intl);
        template <typename T> // function defined in same file
        Tensor(const std::vector<T>& data, bool gradRequired = false);
        static Tensor zeros(const std::vector<size_t>& shape, bool gradRequired = false);
        static Tensor fromMove(const std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {}); // TODO
        static Tensor fromMove(const std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});

        Tensor operator+(Tensor& other);
        Tensor matmul(Tensor& other);
    };
}

// Template functions
template <typename T>
void flattenVecRec(const T val, std::vector<float>& flattened, [[maybe_unused]]std::vector<size_t>& shape, [[maybe_unused]]int depth) {
    flattened.push_back(val);
}
template <typename T>
void flattenVecRec(const std::vector<T>& vec, std::vector<float>& flattened, std::vector<size_t>& shape, int depth) {
    if (depth == static_cast<int>(shape.size())) {
        shape.push_back(vec.size());
    } else {
        if (shape[depth] != vec.size()) throw std::invalid_argument("Tensor: n-dim vector is not rectangular");
    }

    for (T v : vec) {
        flattenVecRec(v, flattened, shape, depth + 1);
    }
}
template <typename T>
inline bassinet::TensorIntl::TensorIntl(const std::vector<T> data, bool gradRequired) {
    std::vector<float> flattened;
    flattenVecRec(data, flattened, _shape, 0);
    _data = std::make_shared<std::vector<float>>(flattened);

    _stride = std::vector<size_t>(_shape.size());
    size_t size{1};
    for (size_t i = _stride.size(); i-- > 0; ) {
        if (i != _stride.size() - 1) _stride[i] = _stride[i + 1];
        _stride[i] = size;
        size *= _shape[i];
    }
    // assert(size == _data.size());
    _gradRequired = gradRequired;
}

template <typename T>
inline bassinet::Tensor::Tensor(const std::vector<T>& data, bool gradRequired) {
    intl = std::make_shared<bassinet::TensorIntl>(bassinet::TensorIntl(data, gradRequired));
}
