#pragma once
#include <memory>
#include <cassert>
#include <string>
#include <atomic>



#include "rand.h"


#include "Type.cuh"
#include "TensorNode.cuh"


static inline std::atomic<long> GLOBAL_TENSOR_ID = 0;


// #define GPU_TO_CPU_WITHOUT_FREE_GPU

class Tensor{
public:
    void* data_;
    void* grad_;

    std::vector<int> shape_;
    std::vector<int> stride_;

    int ndim_;
    std::string name_;
    mutable int version_;
    bool grad_accumulated = false;
    

    // 展平后的数据总量大小
    int size_;
    DType dtype_;
    bool requires_grad_;

    std::shared_ptr<TensorNode> grad_fn_;

public:
    Tensor(const std::vector<int>& shape,DType dtype,bool requires_grad=true,std::string init = "none");
    
    template<typename T>
    Tensor(const std::vector<T>& host_data,DType dtype,bool requires_grad=true,std::string init = "none");

    std::string name()const {return name_;}


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
    void bump_version(){version_++;}

    Tensor operator+(const Tensor& other)const;
    Tensor operator*(const Tensor& other)const;

};


void print_graph(const Tensor& output);