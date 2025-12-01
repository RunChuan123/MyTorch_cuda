#pragma once
#include <memory>
#include <cassert>
#include <string>
#include <atomic>



#include "rand.h"


#include "Type.cuh"
#include "TensorNode.cuh"
#include "TensorStorage.cuh"


static inline std::atomic<long> GLOBAL_TENSOR_ID = 0;


// #define GPU_TO_CPU_WITHOUT_FREE_GPU


class Tensor{
public:

    std::shared_ptr<TensorStorage> storage_;
    std::shared_ptr<TensorNode> grad_fn_;

public:
    Tensor(const std::vector<int>& shape,DType dtype,bool requires_grad=true,std::string init = "none");
    
    template<typename T>
    Tensor(const std::vector<T>& host_data,DType dtype,bool requires_grad=true,std::string init = "none");

    std::string name()const {return storage_->name_;}

    long long size()const {return storage_->size_;}

    DType dtype() const {return storage_->dtype_;}

    const std::vector<int>& shape() const{return storage_->shape_;}


    void zero_grad();

    void backward();

    bool is_leaf() const {
        return grad_fn_->is_leaf;
    }

    void ensure_grad();
    void write_one_to_grad();
    
    
    void print(size_t size=0);
    std::vector<double> cpu_as_double(size_t size =0) const;
    void gpu_copy_from(const void* host_data);

    // 随机初始化 ////////////////
    void uniform_(float from,float to);
    void normal_(float mean,float std);
    void zeros();
    void bump_version(){storage_->version_++;}

    Tensor operator+(const Tensor& other)const;
    Tensor operator*(const Tensor& other)const;

};


void print_graph(const Tensor& output);