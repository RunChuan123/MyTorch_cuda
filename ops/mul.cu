#include "mul.cuh"
#include "../common.cuh"

template<typename T>
__global__ void mul128_kernel(const T* __restrict__ a,const T* __restrict__ b,T* __restrict__ out,int size){
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x*blockDim.x + threadIdx.x)*pack;
    if(offset >= size) return;

    if(offset + pack <= size){
        vec_t va = load128<T>(a+offset);
        vec_t vb = load128<T>(b+offset);
        vec_t vc;
        #pragma unroll
        for(int i=0;i<pack;i++){
            vc[i] = va[i]*vb[i];
        }
        store128(out+offset,vc);
        return;
    }
    for(int i=0;i<pack&&offset+i<size;i++){
        out[offset+i] = a[offset+i] * b[offset+i];
    }
}


template<typename T>
__global__ void mul128_backward_kernel(
    T* __restrict__ grad_a,
    T* __restrict__ grad_b,
    const T* __restrict__ grad_out,
    const T* __restrict__ a,
    const T* __restrict__ b,
    int size
){
    using vec_t = Packed128<T>;
    int pack = vec_t::size;
    int offset = (blockIdx.x*blockDim.x + threadIdx.x)*pack;
    if(offset >= size) return;

    if(offset+pack <= size){
        vec_t go = load128<T>(grad_out+offset);

        vec_t va = load128<T>(a+offset);
        vec_t vb = load128<T>(b+offset);

        vec_t ga = load128<T>(grad_a+offset);
        vec_t gb = load128<T>(grad_b+offset);

        #pragma unroll 
        for(int i=0;i< pack;i++){
            ga[i] += go[i] * vb[i];
            gb[i] += go[i] * vb[i];
        }
        store128(grad_a+offset,ga);
        store128(grad_b+offset,vb);
        return;
    }
    for(int i=0;i<pack&&i+offset<size;i++){
        int k = offset+i;
        grad_a[k] = grad_out[k] * b[k];
        grad_b[k] = grad_out[k] * a[k];
    }
}

Tensor mul(const Tensor& A,const Tensor& B){
    assert(A.storage_->size_ == B.storage_->size_);
    DType dtype = A.storage_->dtype_;

    Tensor out(A.storage_->shape_,dtype,A.storage_->requires_grad_ || B.storage_->requires_grad_);
    DISPATCH_DTYPE_AND_PACK(dtype,scalar_t,vec_t,{
        int pack = vec_t::size;
        int threads = 256;
        int blocks = CEIL_DIV(A.storage_->size_,pack*threads);
        mul128_kernel<scalar_t><<<blocks,threads>>>(
            (scalar_t*)A.storage_->data_,
            (scalar_t*)B.storage_->data_,
            (scalar_t*)out.storage_->data_,
            A.storage_->size_
        );
    });
    if(out.storage_->requires_grad_){
        out.grad_fn_->parents = {const_cast<Tensor*>(&A),const_cast<Tensor*>(&B)};
        out.grad_fn_->op = OpType::Mul;
        out.grad_fn_->ll_args["size"] = {A.storage_->size_};
        out.grad_fn_->tensor_args["A_ref"] = A.storage_;
        out.grad_fn_->tensor_args["B_ref"] = B.storage_;
        out.grad_fn_->int_args["A_version"] = {(int)A.storage_->version_};
        out.grad_fn_->int_args["B_version"] = {(int)B.storage_->version_};
        out.grad_fn_->backward_fn = [&,dtype,size=A.storage_->size_](){
        DISPATCH_DTYPE_AND_PACK(dtype,scalar_t,vec_t,{
            int pack = vec_t::size;
            int threads = 256;
            int blocks = CEIL_DIV(A.storage_->size_,pack*threads);
            mul128_backward_kernel<scalar_t><<<blocks,threads>>>(
            (scalar_t*)A.storage_->grad_,
            (scalar_t*)B.storage_->grad_,
            (scalar_t*)out.storage_->grad_,
            (scalar_t*)A.storage_->data_,
            (scalar_t*)B.storage_->data_,
            size
            );
            });
        };
    }
    return out;
}















// template<int BLOCK_SIZE>
// __global__ void matrix_multiply_shared(const float* A, const float* B, float* C, 
//                                       int M, int N, int K) {
//     // 为每个块分配共享内存
//     __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
//     int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//     int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
//     float sum = 0.0f;
    
//     // 分块矩阵乘法
//     for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
//         // 协作加载数据到共享内存
//         if (row < M && (k * BLOCK_SIZE + threadIdx.x) < K) {
//             As[threadIdx.y][threadIdx.x] = A[row * K + k * BLOCK_SIZE + threadIdx.x];
//         } else {
//             As[threadIdx.y][threadIdx.x] = 0.0f;
//         }
        
//         if ((k * BLOCK_SIZE + threadIdx.y) < K && col < N) {
//             Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * N + col];
//         } else {
//             Bs[threadIdx.y][threadIdx.x] = 0.0f;
//         }
        
//         __syncthreads();  // 等待所有线程完成加载
        
//         // 计算子矩阵乘积
//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
//         }
        
//         __syncthreads();  // 等待所有线程完成计算
//     }
    
//     // 写入结果
//     if (row < M && col < N) {
//         C[row * N + col] = sum;
//     }
// }