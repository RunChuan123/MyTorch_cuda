#pragma once
#include <memory>
#include <cassert>
#include <string>
#include <atomic>

#include "type.cuh"
#include "rand.h"
#include "tensorStorage.cuh"
#include "tensorNode.cuh"


static inline std::atomic<long> GLOBAL_TENSOR_ID = 0;


// #define GPU_TO_CPU_WITHOUT_FREE_GPU


class Tensor{
public:

    std::shared_ptr<TensorStorage> storage_;
    std::shared_ptr<TensorNode> grad_fn_;

public:
    // Tensor({2,2},DType::Float32,true,"none")
    Tensor(const std::vector<int>& shape,DType dtype,bool requires_grad=true,std::string init = "none");
    
    // Tensor(host_ptr,{2,2},DType::x,true)
    Tensor(const float* host_ptr,const std::vector<int>& shape,DType dtype=DType::Float32,bool requires_grad=true);

    // Tensor(1.0f,DType::x,true)
    Tensor(float value ,std::vector<int> shape={1},DType dtype = DType::Float32,bool requires_grad = true);


    Tensor(std::shared_ptr<TensorStorage> storage,std::shared_ptr<TensorNode> grad_fn):
        storage_(storage),grad_fn_(grad_fn){}

    Tensor(){
        storage_ = std::make_shared<TensorStorage>();
    }
    
    

    std::string name()const {return storage_->name_;}

    size_t size()const {return storage_->size_;}

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
    Tensor operator-( float scalar);
    Tensor& operator-=( const Tensor& B);
    Tensor operator-(const Tensor& B);
};


void print_graph(const Tensor& output);