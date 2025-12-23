#include "tensor.cuh"
#include "ops.cuh"

Tensor Tensor::operator+(const Tensor& other) const{
    return add(*this,other);
}
Tensor Tensor::operator*(const Tensor& other) const{
    return mul(*this,other);
}

Tensor Tensor::operator-( const Tensor& B) {
    return sub(*this, B);
}

Tensor Tensor::operator-(float scalar) {
    // 将标量转换为Tensor然后相减
    Tensor scalar_tensor = Tensor(scalar, storage_->shape_, storage_->dtype_);
    return sub(*this, scalar_tensor);
}

Tensor& Tensor::operator-=( const Tensor& B) {
    return sub_(*this, B);
}