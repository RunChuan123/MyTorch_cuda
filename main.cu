#include "mytorch_headers.cuh"
#include "test_utils.cuh"   // 包含上面写的 check_add/check_mul

int main() {
    Tensor t1({2,6}, DType::Float32, true, "rand");
    Tensor t2({2,6}, DType::Float32, true, "rand");
    Tensor t3({2,6}, DType::Float32, true, "rand");

    Tensor mulout =( t1+t2) * t3;

    // 打印
    // mulout.print();
    print_graph(mulout);

    return 0;
}
