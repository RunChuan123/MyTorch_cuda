#include "tensor.cuh"


int main() {
    Tensor a(2);
    Tensor b(4);
    auto c = a + b;
    print_graph(c);
    return 0;
}
