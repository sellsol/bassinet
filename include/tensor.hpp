#include <iostream>
#include <vector>
#include <string>

extern const size_t MAX_TENSOR_PRINT_SIZE;

class Tensor {
private:
    std::vector<float> _data; // will be in row-order
    std::vector<size_t> _shape;
    std::vector<size_t> _stride;

public:
    Tensor(const std::vector<size_t>& shape, float defaultVal = 0);

    const std::vector<size_t>& shape();
    const std::vector<size_t>& stride();
    std::size_t size();

    float& at(const std::vector<size_t>& loc);
    float at(const std::vector<size_t>& loc) const;

    void reshape(const std::vector<size_t>& newShape);
    Tensor operator+(const Tensor& other);
    Tensor matmul(const Tensor& other);

    friend std::ostream& operator<<(std::ostream &os, const Tensor &t);
};