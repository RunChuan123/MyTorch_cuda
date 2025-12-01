#include "tensor.cuh"
#include "utils.cuh"
#include <iostream>
#include "ops/ops.cuh"


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
*  此函数将host_data构造为一维向量
*/
template<typename T>
Tensor::Tensor(const std::vector<T>& host_data,DType dtype,bool requires_grad,std::string init)
    // :size_(host_data.size()),dtype_(dtype),requires_grad_(requires_grad)
{   
    storage_ = std::make_shared<TensorStorage>();
    storage_->shape_ = {(int)host_data.size()};
    storage_->dtype_ = dtype;
    storage_->requires_grad_ = requires_grad;
    storage_->name_ = generate_name();
    storage_->shape_ = {(int)storage_->size_};
    storage_->ndim_ = 1;

    size_t bytes = storage_->size_ * dtype_size(storage_->dtype_);
    cudaMalloc(&storage_->data_,bytes);
    cudaMemcpy(storage_->data_,host_data.data(),bytes,cudaMemcpyHostToDevice);
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


Tensor Tensor::operator+(const Tensor& other) const{
    return add(*this,other);
}
Tensor Tensor::operator*(const Tensor& other) const{
    return mul(*this,other);
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
        scalar_t one = (scalar_t)1;
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
            host[i] = (scalar_t)r;
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
            
            host[i] = (scalar_t)temp[i];
        }
        cudaMemcpy(storage_->data_,host.data(),storage_->size_ * sizeof(scalar_t),cudaMemcpyHostToDevice);
    });
}

void Tensor::zeros(){
    DISPATCH_DTYPE(storage_->dtype_,scalar_t,{
        cudaMemset(storage_->data_,(scalar_t)0,storage_->size_ * sizeof(scalar_t));
    });
}


#include <iostream>
#include <unordered_set>



// static void dfs_node(const Tensor& t,
//                      int depth,
//                      std::unordered_set<const TensorNode*>& visited)
// {
//     TensorNode* node = t.grad_fn_.get();

//     // 防止重复访问
//     if (node && visited.count(node)) return;
//     if (node) visited.insert(node);

//     // 缩进
//     std::string indent(depth * 4, ' ');

//     // 打印当前节点
//     std::cout << indent << "(" << t.name() << ") ";
//     std::cout << "  shape=[";
//     for (int i = 0; i < t.storage_->shape_.size(); i++){
//         std::cout << t.storage_->shape_[i];
//         if(i+1 < t.storage_->shape_.size()) std::cout << ",";
//     }
//     std::cout << "]  dtype=" << (int)t.storage_->dtype_;
//     if(t.storage_->requires_grad_) std::cout << " requires_grad";
//     std::cout << " v" << t.storage_->version_ << "\n";

//     // 叶子就结束
//     if (!node || node->is_leaf) return;

//     // 继续遍历父节点
//     for (const Tensor& p : node->parents)
//         dfs_node(p, depth + 1, visited);
// }

// void print_graph(const Tensor& output) {
//     std::unordered_set<const TensorNode*> visited;
//     std::cout << "=== Autograd Graph ===\n";
//     dfs_node(output, 0, visited);
//     std::cout << "=======================\n";
// }
static void dfs(const Tensor* t,int depth,std::unordered_set<const Tensor*>& visited)
{
    if(!t || visited.count(t))return;
    visited.insert(t);
    // 前缀缩进
    std::string indent(depth * 4,' ');
    std::cout << indent << "(" << t->name() << ") ";
    std::cout << "   shape=[";
    for (int i = 0; i < t->storage_->shape_.size(); i++){
        std::cout << t->storage_->shape_[i];
        if(i+1 < t->storage_->shape_.size()) std::cout << ",";
    }
    std::cout << "]";

    std::cout << "  dtype=" << (int)t->storage_->dtype_;
    if(t->storage_->requires_grad_) std::cout << "  requires_grad";
    std::cout << "  v" << t->storage_->version_;
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