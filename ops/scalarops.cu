#include <cassert>
#include "scalarops.cuh"
#include "../common/common.cuh"
#include "../utils/util.cuh"
#include "../utils/cuda_utils.cuh"


template<typename T>
__global__ void add_kernel(const T* A,const T* B,T* out,int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        out[idx] = A[idx] + B[idx];
    }
}

template<typename T>
__global__ void add128_kernel(const T* __restrict__ A,const T* __restrict__ B,T* __restrict__ out,int size){
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x * blockDim.x + threadIdx.x)*pack;
    
    if(offset >= size)return;

    if(offset + pack <= size){
        vec_t va = load128<T>(A+offset);
        vec_t vb = load128<T>(B+offset);
        vec_t vc;
        #pragma unroll
        for(int i=0;i<pack;i++){
            vc[i] = va[i] + vb[i];
        }
        store128<T>(out + offset,vc);
        return;
    }
    for(int i=0;i< pack&& offset+i <size;i++){
        out[offset + i] = A[offset+i] + B[offset+i];
    }
}

template <typename T>
__global__ void add_inplace_kernel(T* A, const T* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] += B[idx];
    }
}

template<typename T>
__global__ void accumulate128_kernel(T* grad, const T* grad_out, int size)
{
    using vec_t = Packed128<T>;
    int pack = vec_t::size;

    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * pack;

    if (offset >= size) return;

    if (offset + pack <= size) {
        vec_t g  = load128<T>(grad + offset);
        vec_t go = load128<T>(grad_out + offset);

        vec_t out;
        #pragma unroll
        for(int i=0;i<pack;i++){
            out[i] = g[i] + go[i];
        }

        store128<T>(grad + offset, out);
        return;
    }

    for(int i=0;i<pack && offset+i<size;i++){
        grad[offset + i] += grad_out[offset + i];
    }
}




Tensor add(const Tensor& A,const Tensor& B){
    DType dtype = A.storage_->dtype_;
    assert(A.storage_->size_ == B.storage_->size_);

    // int threads = 256;       
    // int blocks = CEIL_DIV(A.size_,threads);

    
    Tensor out = Tensor(A.storage_->shape_,dtype, A.storage_->requires_grad_ || B.storage_->requires_grad_,"none");
    DISPATCH_DTYPE_AND_PACK(dtype,scalar_t,vec_t,{
        int pack = vec_t::size;
        int threads = 256;
        int blocks = CEIL_DIV(A.storage_->size_,pack*threads);
        add128_kernel<scalar_t><<<blocks,threads>>>((scalar_t*)A.storage_->data_,(scalar_t*)B.storage_->data_,(scalar_t*)out.storage_->data_,A.storage_->size_);
    });

    if(out.storage_->requires_grad_){
        out.grad_fn_->parents = {const_cast<Tensor*>(&A),const_cast<Tensor*>(&B)};
        out.grad_fn_->op = OpType::Add;
        out.grad_fn_->sz_args["size"] = {A.storage_->size_};
        out.grad_fn_->tensor_args["A_ref"] = A.storage_;
        out.grad_fn_->tensor_args["B_ref"] = B.storage_;
        out.grad_fn_->int_args["A_version"] = {(int)A.storage_->version_};
        out.grad_fn_->int_args["B_version"] = {(int)B.storage_->version_};
        out.grad_fn_->backward_fn = [&,dtype,size = A.storage_->size_](){
            DISPATCH_DTYPE_AND_PACK(dtype,scalar_t,vec_t,{
                int pack = vec_t::size;
                int threads = 256;
                int blocks = CEIL_DIV(A.storage_->size_,pack*threads);
                accumulate128_kernel<scalar_t><<<blocks,threads>>>((scalar_t*)&A.storage_->grad_,(scalar_t*)&out.storage_->grad_,size);

                accumulate128_kernel<scalar_t><<<blocks,threads>>>((scalar_t*)&B.storage_->grad_,(scalar_t*)&out.storage_->grad_,size);
            })
        };
    }
    return out;
}

//减法

// ===================== 减法kernel =====================

template<typename T>
__global__ void sub_kernel(const T* A, const T* B, T* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = A[idx] - B[idx];
    }
}

template<typename T>
__global__ void sub128_kernel(const T* __restrict__ A, const T* __restrict__ B, 
                              T* __restrict__ out, int size) {
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * pack;
    
    if (offset >= size) return;

    if (offset + pack <= size) {
        vec_t va = load128<T>(A + offset);
        vec_t vb = load128<T>(B + offset);
        vec_t vc;
        #pragma unroll
        for (int i = 0; i < pack; i++) {
            vc[i] = va[i] - vb[i];
        }
        store128<T>(out + offset, vc);
        return;
    }
    
    // 尾部处理
    for (int i = 0; i < pack && offset + i < size; i++) {
        out[offset + i] = A[offset + i] - B[offset + i];
    }
}

template<typename T>
__global__ void sub_inplace_kernel(T* A, const T* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] -= B[idx];
    }
}

// 减法反向传播kernel
template<typename T>
__global__ void sub_backward128_kernel(T* grad_A, T* grad_B, 
                                       const T* grad_out, int size) {
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * pack;

    if (offset >= size) return;

    if (offset + pack <= size) {
        vec_t g_out = load128<T>(grad_out + offset);
        
        // A的梯度: +grad_out
        if (grad_A) {
            vec_t g_A = load128<T>(grad_A + offset);
            vec_t out_A;
            #pragma unroll
            for (int i = 0; i < pack; i++) {
                out_A[i] = g_A[i] + g_out[i];
            }
            store128<T>(grad_A + offset, out_A);
        }
        
        // B的梯度: -grad_out  
        if (grad_B) {
            vec_t g_B = load128<T>(grad_B + offset);
            vec_t out_B;
            #pragma unroll
            for (int i = 0; i < pack; i++) {
                out_B[i] = g_B[i] - g_out[i];
            }
            store128<T>(grad_B + offset, out_B);
        }
        return;
    }

    // 尾部处理
    for (int i = 0; i < pack && offset + i < size; i++) {
        if (grad_A) grad_A[offset + i] += grad_out[offset + i];
        if (grad_B) grad_B[offset + i] -= grad_out[offset + i];
    }
}

// ===================== 主减法函数 =====================


// 专门的B梯度kernel（A-B中的B）
template<typename T>
__global__ void sub_backward_for_B_kernel(T* grad_B, const T* grad_out, int size) {
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * pack;

    if (offset >= size) return;

    if (offset + pack <= size) {
        vec_t g_B = load128<T>(grad_B + offset);
        vec_t g_out = load128<T>(grad_out + offset);
        vec_t out;
        #pragma unroll
        for (int i = 0; i < pack; i++) {
            out[i] = g_B[i] - g_out[i];  // 注意：减去梯度
        }
        store128<T>(grad_B + offset, out);
        return;
    }

    for (int i = 0; i < pack && offset + i < size; i++) {
        grad_B[offset + i] -= grad_out[offset + i];
    }
}

// ===================== 重载操作符 =====================



Tensor sub(const Tensor& A, const Tensor& B) {
    DType dtype = A.storage_->dtype_;
    assert(A.storage_->size_ == B.storage_->size_);
    assert(A.storage_->shape_ == B.storage_->shape_);

    Tensor out = Tensor(A.storage_->shape_, dtype, 
                       A.storage_->requires_grad_ || B.storage_->requires_grad_, 
                       "none");
    
    DISPATCH_DTYPE_AND_PACK(dtype, scalar_t, vec_t, {
        int pack = vec_t::size;
        int threads = 256;
        int blocks = CEIL_DIV(A.storage_->size_, pack * threads);
        
        sub128_kernel<scalar_t><<<blocks, threads>>>(
            (scalar_t*)A.storage_->data_,
            (scalar_t*)B.storage_->data_,
            (scalar_t*)out.storage_->data_,
            A.storage_->size_
        );
    });

    // 自动微分部分
    if (out.storage_->requires_grad_) {
        out.grad_fn_->parents = {const_cast<Tensor*>(&A), const_cast<Tensor*>(&B)};
        out.grad_fn_->op = OpType::Sub;  // 注意：改为Sub操作
        out.grad_fn_->sz_args["size"] = {A.storage_->size_};
        out.grad_fn_->tensor_args["A_ref"] = A.storage_;
        out.grad_fn_->tensor_args["B_ref"] = B.storage_;
        out.grad_fn_->int_args["A_version"] = {(int)A.storage_->version_};
        out.grad_fn_->int_args["B_version"] = {(int)B.storage_->version_};
        
        // 反向传播函数
        out.grad_fn_->backward_fn = [&, dtype, size = A.storage_->size_]() {
            DISPATCH_DTYPE_AND_PACK(dtype, scalar_t, vec_t, {
                int pack = vec_t::size;
                int threads = 256;
                int blocks = CEIL_DIV(size, pack * threads);
                
                // 注意：减法反向传播与加法不同！
                // dL/dA = dL/dC * 1
                // dL/dB = dL/dC * (-1)
                
                // A的梯度：+grad_out
                if(A.storage_->requires_grad_ && B.storage_->requires_grad_){
                    sub_backward128_kernel<scalar_t><<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)B.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        size
                    );
                }else if (A.storage_->requires_grad_) {
                    accumulate128_kernel<scalar_t><<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        size
                    );
                }else{
                    sub_backward_for_B_kernel<scalar_t><<<blocks, threads>>>(
                        (scalar_t*)B.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        size
                    );
                }
            });
        };
    }
    
    return out;
}



// ===================== 就地减法 =====================

Tensor& sub_(Tensor& A, const Tensor& B) {
    assert(A.storage_->size_ == B.storage_->size_);
    assert(A.storage_->dtype_ == B.storage_->dtype_);
    
    DISPATCH_DTYPE_AND_PACK(A.storage_->dtype_, scalar_t, vec_t, {
        int pack = vec_t::size;
        int threads = 256;
        int blocks = CEIL_DIV(A.storage_->size_, pack * threads);
        
        sub_inplace_kernel<scalar_t><<<blocks, threads>>>(
            (scalar_t*)A.storage_->data_,
            (scalar_t*)B.storage_->data_,
            A.storage_->size_
        );
    });
    
    // 就地操作不支持自动微分（会破坏原始数据）
    A.storage_->requires_grad_ = false;
    A.bump_version();
    
    return A;
}

















//求和


// backward kernel: copy scalar grad_out to all positions
template<typename T>
__global__ void sum_backward_kernel(
    T* __restrict__ grad,
    const T* __restrict__ grad_out,  // scalar pointer
    int size)
{
    T g = grad_out[0]; // 所有元素都加这个标量

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        grad[idx] += g;
    }
}



Tensor sum(const Tensor& A){
    DType dtype = A.storage_->dtype_;
    size_t size = A.storage_->size_;

    Tensor out({1},dtype,A.storage_->requires_grad_);
    

    DISPATCH_DTYPE(A.storage_->dtype_,scalar_t,{
        scalar_t* result;
        cudaMalloc(&result,dtype_size(dtype));
        global_sum_deterministic<scalar_t>(result, (scalar_t*)A.storage_->data_,A.storage_->size_,0);
        out.storage_->data_ = result;
    })
    if(out.storage_->requires_grad_){
        out.grad_fn_->parents = {const_cast<Tensor*>(&A)};
        out.grad_fn_->op = OpType::Sum;
        out.grad_fn_->sz_args["size"] = {A.storage_->size_};
        out.grad_fn_->tensor_args["A_ref"] = A.storage_;
        out.grad_fn_->int_args["A_version"] = {(int)A.storage_->version_};
        out.grad_fn_->backward_fn = [&,dtype,size = A.storage_->size_](){
            DISPATCH_DTYPE(dtype, scalar_t, {
                int threads = 256;
                int blocks = CEIL_DIV(size, threads);

                sum_backward_kernel<scalar_t>
                    <<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        size
                );
            });
        };
    }
    return out;
}

template<typename T>
__global__ void sum_dim_kernel128(
    const T* __restrict__ A,
    T* __restrict__ O,
    int outer,     // dim 左边乘积
    int reduce,    // 要 sum 的维度长度
    int inner)     // dim 右边乘积
{
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(inner % pack == 0){
        int size = outer * inner;
        if(idx * pack >= size)return;
        int inner_pack = inner / pack;
        int outer_id = idx / inner_pack;
        int inner_id = idx % inner_pack;

        int inner_start = inner_id * pack;

        vec_t sum_vec = vec_t::zeros();

        int base = outer_id * reduce * inner + inner_start;
        for(int r =0;r < reduce ;r++){
            vec_t v = load128<T>(A + base + r * inner);
            #pragma unroll
            for(int i=0;i< pack;i++){
                sum_vec[i] += v[i];
            }
        }
        int write_pos = outer_id * inner + inner_start;
        store128<T>(O + write_pos,sum_vec);

    }else{
        int pack_left = inner % pack;
        int thread_per_inner = inner / pack + 1;
        if(idx >= thread_per_inner * outer) return;
        int inner_pack = inner / pack +1;
        int outer_id = idx / inner_pack;
        int inner_id = idx % inner_pack;

        int inner_start = inner_id * pack;

        vec_t sum_vec = vec_t::zeros();
        int base = outer_id * reduce * inner + inner_start;
        if(inner_id != inner_pack - 1){
            for(int r=0;r<reduce;r++){
                vec_t v = load128<T>(A + base + r * inner);
                for(int i=0;i< pack;i++){
                    sum_vec[i] += v[i];
                }
            }
        }else{
            
            for(int r=0;r<reduce;r++){
                for(int i=0;i< pack_left;i++){
                    sum_vec[i] += A[base + r * inner + i];
                }
            }
        }
        if(inner_id != inner_pack - 1){
            int write_pos = outer_id * inner + inner_start;
            store128<T>(O + write_pos,sum_vec);
        }else{
            for(int i=0;i<pack_left;i++){
                O[outer_id * inner + inner_start + i] = sum_vec[i];
            }
        }
    }
}



template<typename T>
__global__ void sum_dim_kernel(
    const T* __restrict__ A,
    T* __restrict__ O,
    int outer,
    int reduce,
    int inner){

    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    int total = outer * inner;
    if(idx >= total) return;
    int outer_id = idx / inner;
    int inner_id = idx % inner;
    T sum =0;
    int base = outer_id * reduce * inner + inner_id;
    for(int r =0 ;r< reduce; r++){
        sum+=A[base + r*inner];
    }
    O[idx] = sum;
}


template<typename T>
__global__ void sum_dim_backward_kernel128(
    T* __restrict__ gradA,
    const T* __restrict__ gradO,
    int outer,
    int reduce,
    int inner)
{
    using vec_t = Packed128<T>;
    const int pack = vec_t::size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int inner_vec = inner / pack;          // 完整 pack 数
    int inner_tail = inner % pack;         // 尾部
    int total_vec = outer * inner_vec;     // 向量部分
    int total = total_vec + (inner_tail ? outer : 0);

    if(idx >= total) return;

    if(idx < total_vec)
    {
        // ===== vector path =====
        int outer_id = idx / inner_vec;
        int inner_vec_id = idx % inner_vec;

        int inner_start = inner_vec_id * pack;

        vec_t go = load128<T>(gradO + outer_id * inner + inner_start);

        // 遍历 reduce 次
        for(int r=0; r<reduce; r++){
            T* base = gradA + outer_id * reduce * inner + r * inner + inner_start;

            vec_t ga = load128<T>(base);

            #pragma unroll
            for(int i=0;i<pack;i++){
                ga[i] += go[i];
            }

            store128<T>(base, ga);
        }
    }
    else
    {
        // ===== tail scalar path =====
        int outer_id = idx - total_vec;
        int inner_start = inner_vec * pack;

        // 取 tail 中的 dout
        T go = gradO[outer_id * inner + inner_start];

        for(int r=0; r<reduce; r++){
            T* base = gradA + outer_id * reduce * inner + r * inner + inner_start;

            for(int i=0; i<inner_tail; i++){
                base[i] += go;
            }
        }
    }
}


Tensor sum(const Tensor& A,int dim,bool keep_dim){
    int ndim = A.storage_->shape_.size();
    assert(dim >= 0 && dim < ndim);
    int outer = 1;
    for(int i=0;i<dim;i++){
        outer *= A.storage_->shape_[i];
    }
    int reduce =  A.storage_->shape_[dim];
    int inner = 1;
    for(int i=dim+1;i<ndim;i++){
        inner *= A.storage_->shape_[i];
    }

    // int out_size = outer*inner;
    std::vector<int> out_shape = A.storage_->shape_;
    if(keep_dim){
        out_shape[dim] = 1;
    }else{
        out_shape.erase(out_shape.begin()+dim);
    }
    Tensor out(out_shape,A.storage_->dtype_,A.storage_->requires_grad_,"none");
    DISPATCH_DTYPE_AND_PACK(A.storage_->dtype_,scalar_t,vec_t,{
        int pack = vec_t::size;
        int total_vec = (outer * inner) / pack;
        bool has_tail = (inner % pack != 0);
        int total = total_vec + (has_tail ? outer : 0);
        int threads = 256;
        int blocks = CEIL_DIV(total,threads);
        sum_dim_kernel128<<<blocks,threads>>>(
            (scalar_t*)A.storage_->data_,
            (scalar_t*)out.storage_->data_,
            outer, reduce, inner);
    });
    if(out.storage_->requires_grad_){
        out.grad_fn_->parents = { const_cast<Tensor*>(&A) };
        out.grad_fn_->op = OpType::SumDim;
        out.grad_fn_->sz_args["outer"] = outer;
        out.grad_fn_->sz_args["reduce"] = reduce;
        out.grad_fn_->sz_args["inner"] = inner;

        out.grad_fn_->backward_fn = [&, dtype=A.dtype()]() {
            DISPATCH_DTYPE(dtype, scalar_t, {

                int pack = Packed128<scalar_t>::size;

                int total_vec = (outer * inner) / pack;
                int has_tail = (inner % pack != 0);
                int total = total_vec + (has_tail ? outer : 0);

                int threads = 256;
                int blocks = CEIL_DIV(total, threads);

                sum_dim_backward_kernel128<scalar_t>
                    <<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        outer, reduce, inner
                    );
            });
        };
    }
    return out;  
}

// mean

template<typename T>
__global__ void mean_backward_kernel(
    T* __restrict__ gradA,
    const T* __restrict__ gradO,
    int size)
{
    T g = gradO[0] / (T)size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        gradA[idx] += g;
    }
}


Tensor mean(const Tensor& A)
{
    DType dtype = A.storage_->dtype_;
    int size = A.storage_->size_;

    // forward = sum(A) / size
    Tensor out = sum(A);
    DISPATCH_DTYPE(dtype, scalar_t, {
        scalar_t val;
        cudaMemcpy(&val, out.storage_->data_, sizeof(scalar_t), cudaMemcpyDeviceToHost);
        val = val / (scalar_t)size;
        cudaMemcpy(out.storage_->data_, &val, sizeof(scalar_t), cudaMemcpyHostToDevice);
    });

    // backward
    if(out.storage_->requires_grad_) {
        out.grad_fn_->parents = { const_cast<Tensor*>(&A) };
        out.grad_fn_->op = OpType::Mean;
        out.grad_fn_->sz_args["size"] = size;

        out.grad_fn_->backward_fn = [&, dtype=size]() {
            DISPATCH_DTYPE(A.storage_->dtype_, scalar_t, {
                int threads = 256;
                int blocks = CEIL_DIV(size, threads);

                mean_backward_kernel<scalar_t>
                    <<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        size
                    );
            });
        };
    }

    return out;
}

template<typename T>
__global__ void mean_dim_backward_kernel128(
    T* __restrict__ gradA,
    const T* __restrict__ gradO,
    int outer,
    int reduce,
    int inner)
{
    using vec_t = Packed128<T>;
    int pack = vec_t::size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int inner_vec = inner / pack;
    int inner_tail = inner % pack;

    int total_vec = outer * inner_vec;
    int total = total_vec + (inner_tail ? outer : 0);

    if(idx >= total) return;

    if(idx < total_vec)
    {
        int outer_id = idx / inner_vec;
        int inner_vec_id = idx % inner_vec;

        int inner_start = inner_vec_id * pack;

        vec_t go = load128<T>(gradO + outer_id * inner + inner_start);

        #pragma unroll
        for(int i=0;i<pack;i++){
            go[i] /= (T)reduce;
        }

        for(int r=0; r<reduce; r++){
            T* base = gradA + outer_id * reduce * inner + r * inner + inner_start;

            vec_t ga = load128<T>(base);
            #pragma unroll
            for(int i=0;i<pack;i++){
                ga[i] += go[i];
            }
            store128(base, ga);
        }
    }
    else
    {
        int outer_id = idx - total_vec;
        int inner_start = inner_vec * pack;

        T go = gradO[outer_id * inner + inner_start] / (T)reduce;

        for(int r=0; r<reduce; r++){
            T* base = gradA + outer_id * reduce * inner + r * inner + inner_start;

            for(int i=0;i<inner_tail;i++){
                base[i] += go;
            }
        }
    }
}
inline void compute_reduce_geometry(
    const std::vector<int>& shape,
    int dim,
    int& outer,
    int& reduce,
    int& inner)
{
    assert(dim >= 0 && dim < (int)shape.size());

    outer = 1;
    for(int i = 0; i < dim; i++)
        outer *= shape[i];

    reduce = shape[dim];

    inner = 1;
    for(int i = dim + 1; i < (int)shape.size(); i++)
        inner *= shape[i];
}


Tensor mean(const Tensor& A, int dim)
{
    // 计算 (outer, reduce, inner)
    int outer, reduce, inner;
    compute_reduce_geometry(A.storage_->shape_, dim, outer, reduce, inner);

    // 构造输出 shape
    auto out_shape = A.storage_->shape_;
    out_shape.erase(out_shape.begin() + dim);

    // Tensor out(out_shape, A.dtype(), A.storage_->requires_grad_);

    // // forward = sum(dim) / reduce
    // Tensor tmp = sum(A, dim);

    // DISPATCH_DTYPE(A.dtype(), scalar_t, {
    //     int size = tmp.storage_->size_;
    //     std::vector<scalar_t> host(size);
    //     cudaMemcpy(host.data(), tmp.storage_->data_, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    //     for(int i=0;i<size;i++) host[i] /= (scalar_t)reduce;
    //     cudaMemcpy(out.storage_->data_, host.data(), size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    // });

        // Tensor out(out_shape, A.dtype(), A.storage_->requires_grad_);

    // forward = sum(dim) / reduce
    Tensor out = sum(A, dim);

    // DISPATCH_DTYPE(A.dtype(), scalar_t, {
    //     int size = tmp.storage_->size_;
    //     std::vector<scalar_t> host(size);
    //     cudaMemcpy(host.data(), tmp.storage_->data_, size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    //     for(int i=0;i<size;i++) host[i] /= (scalar_t)reduce;
    //     cudaMemcpy(out.storage_->data_, host.data(), size * sizeof(scalar_t), cudaMemcpyHostToDevice);
    // });


    // backward
    if(out.storage_->requires_grad_) {
        out.grad_fn_->parents = { const_cast<Tensor*>(&A) };
        out.grad_fn_->op = OpType::MeanDim;

        out.grad_fn_->sz_args["outer"] = outer;
        out.grad_fn_->sz_args["reduce"] = reduce;
        out.grad_fn_->sz_args["inner"] = inner;

        out.grad_fn_->backward_fn = [&, dtype=A.dtype()]() {
            DISPATCH_DTYPE(dtype, scalar_t, {

                int pack = Packed128<scalar_t>::size;
                int inner_vec = inner / pack;
                int has_tail = inner % pack != 0;

                int total = outer * inner_vec + (has_tail ? outer : 0);

                int threads = 256;
                int blocks = CEIL_DIV(total, threads);

                mean_dim_backward_kernel128<scalar_t>
                    <<<blocks, threads>>>(
                        (scalar_t*)A.storage_->grad_,
                        (scalar_t*)out.storage_->grad_,
                        outer, reduce, inner
                    );
            });
        };
    }

    return out;
}


// Tensor mean(const Tensor&A){
//     return Tensor(1.0f);

// }
// Tensor mean(const Tensor& A,int dim){
//     return Tensor(1.0f);

// }

