#include "add.cuh"
#include "../common.cuh"
#include "../utils.cuh"


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
        out.grad_fn_->ll_args["size"] = {A.storage_->size_};
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