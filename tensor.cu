#include "tensor.cuh"
#include "utils.cuh"
#include <iostream>
#include "ops/ops.cuh"


static inline std::string generate_name(const std::string& prefix = "tensor"){
    long id = GLOBAL_TENSOR_ID.fetch_add(1);
    return prefix + "_" + std::to_string(id);
}

Tensor::Tensor(const std::vector<int>& shape,DType dtype,bool requires_grad,std::string init):
    shape_(shape),dtype_(dtype),requires_grad_(requires_grad)
{
    name_ = generate_name();
    ndim_ = shape_.size();
    stride_ = compute_stride(shape_);
    size_ = 1;
    for(int dim: shape_) size_ *= dim;

    size_t bytes = size_ * dtype_size(dtype_);
    cudaMalloc(&data_,bytes);
    grad_ = nullptr;
    grad_fn_ = std::make_shared<TensorNode>(this);
    grad_fn_->is_leaf = true;

    if(init == "rand"){
        uniform_(0.0f,1.0f);
    }else if (init == "randn")
    {
        normal_(0.0f,1.0f);
    }else{
        zeros();
    }
    
}


/*
*  此函数将host_data构造为一维向量
*/
template<typename T>
Tensor::Tensor(const std::vector<T>& host_data,DType dtype,bool requires_grad,std::string init):
    size_(host_data.size()),dtype_(dtype),requires_grad_(requires_grad)
{   
    name_ = generate_name();
    shape_ = {(int)size_};
    ndim_ = 1;

    size_t bytes = size_ * dtype_size(dtype_);
    cudaMalloc(&data_,bytes);
    cudaMemcpy(data_,host_data.data(),bytes,cudaMemcpyHostToDevice);
    grad_ = nullptr;
    grad_fn_ = std::make_shared<TensorNode>(this);
    grad_fn_->is_leaf = true;
    

    if(init == "rand"){
        uniform_(0.0f,1.0f);
    }else if (init == "randn")
    {
        normal_(0.0f,1.0f);
    }else{
        zeros();
    }
}


Tensor Tensor::operator+(const Tensor& other) const{
    return add(*this,other);
}
Tensor Tensor::operator*(const Tensor& other) const{
    return mul(*this,other);
}

void Tensor::zero_grad(){
    assert(requires_grad_);
    cudaMemset(grad_,0,size_ * dtype_size(dtype_));
}

void Tensor::ensure_grad(){
    assert(requires_grad_);
    if (grad_ == nullptr){
        cudaMalloc(&grad_,size_*dtype_size(dtype_));
        // write_one_to_grad();
    }
}

void Tensor::write_one_to_grad() {
    DISPATCH_DTYPE(dtype_, scalar_t, {
        scalar_t one = (scalar_t)1;
        cudaMemcpy(grad_, &one, sizeof(scalar_t), cudaMemcpyHostToDevice);
    });
}


// 递归调用backward
void Tensor::backward(){
    assert(size_ == 1  && "test if this message will print");
    
    if(!requires_grad_) return;
    
    ensure_grad();
    write_one_to_grad();

    std::function<void(TensorNode*)> dfs = [&](TensorNode* fn){
        if(fn->backward_fn)fn->backward_fn();
        for(Tensor* p : fn-> parents){
            if(p->requires_grad_){
                p->ensure_grad();
                dfs(p->grad_fn_.get());
            }
        }
    };

    dfs(grad_fn_.get());
}


void Tensor::print(size_t size){
    int need_size = size ? size : size_ ;
    std::cout << "Tensor(shape=[";
    auto cpu_data = cpu_as_double(need_size);
    for (int i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ",";
    }
    std::cout << "], dtype=" << dtype_ << "):\n";


    for (int i = 0; i < need_size; i++) {
        std::cout << cpu_data[i] << " ";
    }
    std::cout << "\n";
}

template<typename scalar_t>
std::vector<scalar_t> _tensor_to_cpu(const Tensor* t,size_t size) {
    std::vector<scalar_t> cpu_data(t->size_);
    int need_move_size = size ? size : t->size_ ;
    cudaMemcpy(cpu_data.data(),
               t->data_,
               need_move_size * sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
    return cpu_data;
}


std::vector<double> Tensor::cpu_as_double(size_t size)const {
    
    int need_move_size = size ? size : size_ ;
    std::vector<double> out(need_move_size);
    DISPATCH_DTYPE(dtype_, scalar_t, {
        std::vector<scalar_t> tmp(need_move_size);
        cudaMemcpy(tmp.data(), data_, need_move_size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < need_move_size; i++)
            out[i] = (double)tmp[i];
    });
    return out;
}

void Tensor::gpu_copy_from(const void* host_data) {
    cudaMemcpy(data_, host_data, size_ * dtype_size(dtype_), cudaMemcpyHostToDevice);
}


void Tensor::uniform_(float from,float to){
    DISPATCH_DTYPE(dtype_,scalar_t,{
        std::vector<scalar_t> host(size_);
        for(int i=0;i<size_;i++){
            float r = randfloat32(&GLOBAL_RNG) * (to-from) + from;
            host[i] = (scalar_t)r;
        }
        cudaMemcpy(data_,host.data(),size_ * sizeof(scalar_t),cudaMemcpyHostToDevice);
    });
}

void Tensor::normal_(float mean,float std){
    DISPATCH_DTYPE(dtype_,scalar_t,{
        std::vector<float> temp(size_);
        std::vector<float> host(size_);
        ::normal_(temp.data(),size_,mean,std,&GLOBAL_RNG);
        for(int i=0;i<size_;i++){
            
            host[i] = (scalar_t)temp[i];
        }
        cudaMemcpy(data_,host.data(),size_ * sizeof(scalar_t),cudaMemcpyHostToDevice);
    });
}

void Tensor::zeros(){
    DISPATCH_DTYPE(dtype_,scalar_t,{
        cudaMemset(data_,(scalar_t)0,size_ * sizeof(scalar_t));
    });
}


#include <iostream>
#include <string>
#include <unordered_set>



static void dfs(const Tensor* t,int depth,std::unordered_set<const Tensor*>& visited)
{
    if(!t || visited.count(t))return;
    visited.insert(t);
    // 前缀缩进
    std::string indent(depth * 4,' ');
    std::cout << indent << "(" << t->name() << ") ";
    std::cout << "   shape=[";
    for (int i = 0; i < t->shape_.size(); i++){
        std::cout << t->shape_[i];
        if(i+1 < t->shape_.size()) std::cout << ",";
    }
    std::cout << "]";

    std::cout << "  dtype=" << (int)t->dtype_;
    if(t->requires_grad_) std::cout << "  requires_grad";
    std::cout << "  v" << t->version_;
    std::cout << std::endl;

    if(!t->grad_fn_) return;  // leaf节点没有parents

    for (Tensor* p : t->grad_fn_->parents)
        dfs(p, depth + 1, visited);
}

void print_graph(const Tensor& output){
    std::unordered_set<const Tensor*> visited;
    std::cout << "=== Autograd Graph ===\n";
    dfs(&output, 0, visited);
    std::cout << "=======================\n";
}