__global__ void reduction_kernel(const float* input, float* output, int n) {
    __shared__ float partial_sums[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    partial_sums[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 在共享内存中进行树状归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    // 第一个线程写入结果
    if (tid == 0) {
        output[blockIdx.x] = partial_sums[0];
    }
}