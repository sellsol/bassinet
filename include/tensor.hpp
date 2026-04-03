#pragma once

#include <iostream>
#include <vector>

extern const std::size_t MAX_TENSOR_PRINT_SIZE;

namespace bassinet {

    class TensorOp;


    class Tensor: public std::enable_shared_from_this<Tensor> {
    private:
        std::shared_ptr<std::vector<float>> _data; // will be in row-order
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;

        bool _gradRequired;
        std::vector<float> _grad; // if access by tensor index required, just use same calcs as for _data I think??
        std::function<void(std::vector<Tensor*>&, Tensor&)> _gradFn; // set by forward function version
        std::vector<Tensor*> _parents;

    public:
        Tensor(const std::vector<std::size_t>& shape, float defaultVal = 0.0f, bool gradRequired = false);
        Tensor(std::vector<float> data, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& stride, bool gradRequired = false, std::function<void(std::vector<Tensor*>&, Tensor&)> gradFn = nullptr, const std::vector<Tensor*>& parents = {});
        Tensor(std::shared_ptr<std::vector<float>> data, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& stride, bool gradRequired = false, std::function<void(std::vector<Tensor*>&, Tensor&)> gradFn = nullptr, const std::vector<Tensor*>& parents = {});

        const std::shared_ptr<std::vector<float>>& rawData() const;
        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        std::size_t size() const;

        bool gradRequired() const;
        const std::vector<float>& grad() const;

        float& at(const std::vector<std::size_t>& loc);
        float at(const std::vector<std::size_t>& loc) const;
        Tensor transpose(std::size_t dim0, std::size_t dim1) const;
        // Tensor flatten(std::size_t start, std::size_t end) const;

        void addToData(const std::vector<float>& additions);
        void zeroGrad();
        void addToGrad(const std::vector<float>& additions);
        void backward();

        Tensor operator+(Tensor& other);
        Tensor matmul(Tensor& other);
    };

    std::ostream& operator<<(std::ostream& out, const Tensor& t); // namespace resolved through ADL

    class TensorOp {
    public:
        virtual Tensor forward(const std::vector<Tensor*>& parents) = 0;
        virtual void backward(const std::vector<Tensor*>& parents, Tensor& child) = 0;
    };
}