#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__global__ void softmax_baseline(const T* __restrict__ in,
                                 T* __restrict__ out,
                                 int N)   // N = row length
{
    // one block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // dynamic shared memory for reduction
    extern __shared__ T buf[];  // size = blockDim.x

    // -------- 1. compute row max --------
    T local_max = -INFINITY;

    for (int i = tid; i < N; i += blockDim.x) {
        T v = in[row * N + i];
        local_max = v > local_max ? v : local_max;
    }

    buf[tid] = local_max;
    __syncthreads();

    // block reduction (max)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            T other = buf[tid + stride];
            buf[tid] = buf[tid] > other ? buf[tid] : other;
        }
        __syncthreads();
    }

    T row_max = buf[0];
    __syncthreads();

    // -------- 2. compute sum(exp(x - max)) --------
    T local_sum = 0;

    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += exp(in[row * N + i] - row_max);
    }

    buf[tid] = local_sum;
    __syncthreads();

    // block reduction (sum)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }

    T row_sum = buf[0];
    __syncthreads();

    // -------- 3. write output --------
    for (int i = tid; i < N; i += blockDim.x) {
        out[row * N + i] =
            exp(in[row * N + i] - row_max) / row_sum;
    }
}
