#include <vector>
#include <string>

#include "Type.cuh"


struct TensorStorage{
    void* data_ = nullptr;
    void* grad_ = nullptr;

    std::vector<int> shape_;
    std::vector<int> stride_;
    
    int ndim_;
    std::string name_;
    mutable int version_;
    bool grad_accumulated = false;
    // 展平后的数据总量大小
    long long size_;
    DType dtype_;
    bool requires_grad_;
};