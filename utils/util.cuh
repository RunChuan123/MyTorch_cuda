#pragma once
#include <vector>
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

inline void seed(uint64_t s){
    manual_seed(&GLOBAL_RNG,s);
}

// template<typename scalar_t>
// void fill_on_cpu(std::vector<scalar_t>& host,int size,std::function<scalar_t()> generator){
//     for(int i=0;i<size;i++){
//         host[i] = generator();
//     }
// }




