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
        TensorIntl(const std::vector<size_t>& shape, float defaultVal = 0.0f, bool gradRequired = false);
        TensorIntl(std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});
        TensorIntl(std::shared_ptr<std::vector<float>> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});

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

        Tensor(const std::vector<size_t>& shape, float defaultVal = 0.0f, bool gradRequired = false);
        Tensor(std::vector<float> data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});
        Tensor(std::shared_ptr<std::vector<float>>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, bool gradRequired = false, std::function<void(std::vector<std::shared_ptr<TensorIntl>>&, TensorIntl&)> gradFn = nullptr, const std::vector<std::shared_ptr<TensorIntl>>& parents = {});
        Tensor(std::shared_ptr<TensorIntl> intl);

        Tensor operator+(Tensor& other);
        Tensor matmul(Tensor& other);
    };
}