#pragma once
#include <vector>
#include <cstdint>

#include "rand.h"


inline std::vector<int> compute_stride(const std::vector<int>& shape){
    int ndim = shape.size();
    std::vector<int> stride(ndim);
    stride[ndim-1] = 1;
    for(int i = ndim-2 ;i >= 0;i--){
        stride[i] = stride[i+1] * shape[i+1];
    }
    return stride;
}


static mt19937_state GLOBAL_RNG;





