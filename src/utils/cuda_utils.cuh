#pragma once

#include "type.cuh"

#define WARP_SIZE 32 

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, "  Code: %d, Reason: %s\n", err, cudaGetErrorString(err)); \
        fprintf(stderr, "  Function: %s\n", #call); \
        exit(EXIT_FAILURE); \
    } \
}while(0)


__device__ inline float warpReduceSum(float val){
    for(int offset = 16;offset > 0;offset/=2){
        val += __shfl_xor_sync(0xffffffff,val,offset);
    }
    return val;
}

__device__ inline float warpReduceMax(float val){
    for(int offset = 16;offset > 0;offset/=2){
        val =fmaxf(val, __shfl_xor_sync(0xffffffff,val,offset));
    }
    return val;
}


using reduction_func_t = float (*)(float);

/*
* 在循环中使用时需要开启final_sync，防止破坏共享内存状态
*/
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val,bool final_sync = false,float out_of_bound = 0.0f){
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if(lane_id == 0) {shared_val[warp_id] = warp_val;}
    __syncthreads();
    warp_val = (lane_id < num_warps)? shared_val[lane_id] : out_of_bound;
    float block_val = warp_reduction(warp_val);

    if(final_sync){
        __syncthreads();
    }
    return block_val;
}


template<class Float>
__global__ void global_sum_single_block_kernel(Float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = static_cast<Float>(reduction);
    }
}

template<class Float>
void global_sum_deterministic(Float* result, const Float* values, size_t count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    CUDA_CHECK(cudaGetLastError());
}


inline void cuda_convet_data_with_diff_type(void* dst,const float* src,DType dtype,size_t size){
    DISPATCH_DTYPE(dtype,scalar_t,{
        scalar_t* tmp = (scalar_t*)malloc(size*dtype_size(dtype));
        for(int i=0;i<size;i++){
            tmp[i] = static_cast<scalar_t>(src[i]);
        }
        cudaMemcpy(dst,tmp,size*dtype_size(dtype),cudaMemcpyHostToDevice);
        free(tmp);
    })
}


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
        fprintf(stderr, "  Code: %d, Reason: %s\n", err, cudaGetErrorString(err)); \
        fprintf(stderr, "  Function: %s\n", #call); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

#define CEIL_DIV(M,N) ((M+N-1)/M)


