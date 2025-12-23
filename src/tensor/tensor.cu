#include <iostream>

#include "tensor.cuh"
// #include "ops.cuh"
#include "util.h"
#include "cuda_utils.cuh"
#include "rand.h"


static inline std::string generate_name(const std::string& prefix = "tensor"){
    long id = GLOBAL_TENSOR_ID.fetch_add(1);
    return prefix + "_" + std::to_string(id);
}



Tensor::Tensor(const std::vector<int>& shape,DType dtype,bool requires_grad,std::string init)
    // :shape_(shape),dtype_(dtype),requires_grad_(requires_grad)
{
    storage_ = std::make_shared<TensorStorage>();
    storage_->shape_ = shape;
    storage_->dtype_ = dtype;
    storage_->requires_grad_ = requires_grad;
    storage_->name_ = generate_name();
    storage_->ndim_ = storage_->shape_.size();
    storage_->stride_ = compute_stride(storage_->shape_);
    storage_->size_ = 1;
    for(int dim: storage_->shape_) storage_->size_ *= dim;

    size_t bytes = storage_->size_ * dtype_size(storage_->dtype_);
    cudaMalloc(&storage_->data_,bytes);
    storage_->grad_ = nullptr;
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
*  暂时针对float*和DType::Float32类型的数据进行转换
* 
*/

Tensor::Tensor(const float* host_ptr,const std::vector<int>& shape,DType dtype,bool requires_grad){   
    storage_ = std::make_shared<TensorStorage>();
    storage_->shape_ = shape;
    storage_->dtype_ = dtype;
    storage_->requires_grad_ = requires_grad;
    storage_->name_ = generate_name();
    storage_->ndim_ = storage_->shape_.size();
    storage_->stride_ = compute_stride(storage_->shape_);
    storage_->size_ = 1;
    for(int dim: storage_->shape_) storage_->size_ *= dim;

    size_t bytes = storage_->size_ * dtype_size(storage_->dtype_);

    cudaMalloc(&storage_->data_,bytes);

    cuda_convet_data_with_diff_type(storage_->data_,host_ptr,dtype,storage_->size_);

    
    storage_->grad_ = nullptr;
    grad_fn_ = std::make_shared<TensorNode>(this);
    grad_fn_->is_leaf = true;

}




Tensor::Tensor(float value ,std::vector<int> shape,DType dtype,bool requires_grad){
    storage_ = std::make_shared<TensorStorage>();
    storage_->shape_ = shape;
    storage_->dtype_ = dtype;
    storage_->requires_grad_ = requires_grad;
    storage_->name_ = generate_name();
    storage_->size_ = 1;
    for(auto& i: shape){storage_->size_*i;}
    storage_->ndim_ = shape.size();
    cudaMalloc(&storage_->data_,dtype_size(dtype)*storage_->size_);
    cudaMemcpy(storage_->data_,&value,dtype_size(dtype)*storage_->size_,cudaMemcpyHostToDevice);
    storage_->grad_ = nullptr;
    grad_fn_ = std::make_shared<TensorNode>(this);
    grad_fn_->is_leaf = true;
}




void Tensor::zero_grad(){
    assert(storage_->requires_grad_);
    cudaMemset(storage_->grad_,0,storage_->size_ * dtype_size(storage_->dtype_));
}

void Tensor::ensure_grad(){
    assert(storage_->requires_grad_);
    if (storage_->grad_ == nullptr){
        cudaMalloc(&storage_->grad_,storage_->size_*dtype_size(storage_->dtype_));
        // write_one_to_grad();
    }
}

void Tensor::write_one_to_grad() {
    DISPATCH_DTYPE(storage_->dtype_, scalar_t, {
        scalar_t one = static_cast<scalar_t>(1.0f);
        cudaMemcpy(storage_->grad_, &one, sizeof(scalar_t), cudaMemcpyHostToDevice);
    });
}


// 递归调用backward
void Tensor::backward(){
    assert(storage_->size_ == 1  && "test if this message will print");
    
    if(!storage_->requires_grad_) return;
    
    ensure_grad();
    write_one_to_grad();

    std::function<void(TensorNode*)> dfs = [&](TensorNode* fn){
        if(fn->backward_fn)fn->backward_fn();
        for(Tensor* p : fn-> parents){
            if(p->storage_->requires_grad_){
                p->ensure_grad();
                dfs(p->grad_fn_.get());
            }
        }
    };

    dfs(grad_fn_.get());
}


void Tensor::print(size_t size){
    int need_size = size ? size : storage_->size_ ;
    std::cout << "Tensor(shape=[";
    auto cpu_data = cpu_as_double(need_size);
    for (int i = 0; i < storage_->shape_.size(); i++) {
        std::cout << storage_->shape_[i];
        if (i + 1 < storage_->shape_.size()) std::cout << ",";
    }
    std::cout << "], dtype=" << storage_->dtype_ << "):\n";


    for (int i = 0; i < need_size; i++) {
        std::cout << cpu_data[i] << " ";
    }
    std::cout << "\n";
}

template<typename scalar_t>
std::vector<scalar_t> _tensor_to_cpu(const Tensor* t,size_t size) {
    std::vector<scalar_t> cpu_data(t->storage_->size_);
    int need_move_size = size ? size : t->storage_->size_ ;
    cudaMemcpy(cpu_data.data(),
               t->storage_->data_,
               need_move_size * sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
    return cpu_data;
}


std::vector<double> Tensor::cpu_as_double(size_t size)const {
    
    int need_move_size = size ? size : storage_->size_ ;
    std::vector<double> out(need_move_size);
    DISPATCH_DTYPE(storage_->dtype_, scalar_t, {
        std::vector<scalar_t> tmp(need_move_size);
        cudaMemcpy(tmp.data(), storage_->data_, need_move_size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < need_move_size; i++)
            out[i] = (double)tmp[i];
    });
    return out;
}

void Tensor::gpu_copy_from(const void* host_data) {
    cudaMemcpy(storage_->data_, host_data, storage_->size_ * dtype_size(storage_->dtype_), cudaMemcpyHostToDevice);
}


void Tensor::uniform_(float from,float to){
    DISPATCH_DTYPE(storage_->dtype_,scalar_t,{
        std::vector<scalar_t> host(storage_->size_);
        for(int i=0;i<storage_->size_;i++){
            float r = randfloat32(&GLOBAL_RNG) * (to-from) + from;
            host[i] = static_cast<scalar_t>(r);
        }
        cudaMemcpy(storage_->data_,host.data(),storage_->size_ * sizeof(scalar_t),cudaMemcpyHostToDevice);
    });
}

void Tensor::normal_(float mean,float std){
    DISPATCH_DTYPE(storage_->dtype_,scalar_t,{
        std::vector<float> temp(storage_->size_);
        std::vector<float> host(storage_->size_);
        ::normal_(temp.data(),storage_->size_,mean,std,&GLOBAL_RNG);
        for(int i=0;i<storage_->size_;i++){
            
            host[i] = static_cast<scalar_t>(temp[i]);
        }
        cudaMemcpy(storage_->data_,host.data(),storage_->size_ * sizeof(scalar_t),cudaMemcpyHostToDevice);
    });
}

void Tensor::zeros(){
    DISPATCH_DTYPE(storage_->dtype_,scalar_t,{
        cudaMemset(storage_->data_,static_cast<scalar_t>(0.0f),storage_->size_ * sizeof(scalar_t));
    });
}


#include <iostream>
#include <unordered_set>




// static void dfs(const Tensor* t,int depth,std::unordered_set<const Tensor*>& visited)
// {
//     if(!t || visited.count(t))return;
//     visited.insert(t);
//     // 前缀缩进
//     std::string indent(depth * 4,' ');
//     std::cout << indent << "(" << t->name() << ") ";
//     std::cout << "   shape=[";
//     for (int i = 0; i < t->storage_->shape_.size(); i++){
//         std::cout << t->storage_->shape_[i];
//         if(i+1 < t->storage_->shape_.size()) std::cout << ",";
//     }
//     std::cout << "]";

//     std::cout << "  dtype=" << (int)t->storage_->dtype_;
//     if(t->storage_->requires_grad_) std::cout << "  requires_grad";
//     std::cout << "  v" << t->storage_->version_;
//     std::cout << std::endl;

//     if(!t->grad_fn_) return;  // leaf节点没有parents

//     for (Tensor* p : t->grad_fn_->parents)
//         dfs(p, depth + 1, visited);
// }

// void print_graph(const Tensor& output){
//     std::unordered_set<const Tensor*> visited;
//     std::cout << "=== Autograd Graph ===\n";
//     dfs(&output, 0, visited);
//     std::cout << "=======================\n";
// }
static void dfs_tree(const Tensor* t,
                     const std::string& prefix,
                     bool is_last,
                     std::unordered_set<const Tensor*>& visited)
{
    if (!t || visited.count(t)) return;
    visited.insert(t);

    // ├── 或 └──
    std::cout << prefix 
              << (is_last ? "└── " : "├── ")
              << "(" << t->name() << ") ";

    std::cout << "shape=[";
    for (size_t i = 0; i < t->storage_->shape_.size(); i++){
        std::cout << t->storage_->shape_[i];
        if (i + 1 < t->storage_->shape_.size()) std::cout << ",";
    }
    std::cout << "]";

    std::cout << " dtype=" << t->storage_->dtype_;
    if (!t->storage_->requires_grad_) std::cout << " no grad";
    std::cout << " v" << t->storage_->version_;
    std::cout << t->grad_fn_->op;  
    std::cout << "\n";

    // 叶子节点
    if (!t->grad_fn_) return;

    auto& parents = t->grad_fn_->parents;
    for (size_t i = 0; i < parents.size(); i++){
        bool last = (i == parents.size() - 1);
        dfs_tree(
            parents[i],
            prefix + (is_last ? "    " : "│   "),
            last,
            visited
        );
    }
}

void print_graph(const Tensor& output){
    std::unordered_set<const Tensor*> visited;
    std::cout << "=== Autograd Graph ===\n";
    dfs_tree(&output, "", true, visited);
    std::cout << "=======================\n";
}
