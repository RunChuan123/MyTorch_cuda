#include <vector>
#include <cmath>
#include <iostream>
#include "tensor.cuh"

bool almost_equal(double a, double b, double eps = 1e-5) {
    return std::fabs(a - b) < eps;
}

bool check_add( Tensor& A,  Tensor& B,  Tensor& out) {
    auto a = A.cpu_as_double();
    auto b = B.cpu_as_double();
    auto o = out.cpu_as_double();

    bool ok = true;
    for (int i = 0; i < A.storage_->size_; i++) {
        double expected = a[i] + b[i];
        if (!almost_equal(o[i], expected)) {
            std::cout << "[ADD ERROR] idx " << i 
                      << ": got=" << o[i]
                      << " but expected=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}


bool check_mul(const Tensor& A, const Tensor& B, const Tensor& out) {
    auto a = A.cpu_as_double();
    auto b = B.cpu_as_double();
    auto o = out.cpu_as_double();

    bool ok = true;
    for (int i = 0; i < A.storage_->size_; i++) {
        double expected = a[i] * b[i];
        if (!almost_equal(o[i], expected)) {
            std::cout << "[MUL ERROR] idx " << i 
                      << ": got=" << o[i]
                      << " but expected=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}
