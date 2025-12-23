
#define CEIL_DIV(M,N) ((M+N-1)/N)


// block 2D (block_size,block_size)
template<typename T>
__global__ void mat_maul_v1(T* __restrict__ data_1,T* __restrict__ data_2,T* __restrict__ data_3,int M,int N,int K){
    // M*K,K*N,M*N
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if(idx > M*N)return;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= M || col >= N)return;
    T tmp = 0.0f;
    for(int k =0;k< K;k++){
        tmp += data_1[row * K + k] * data_2[k*N + col];
    }
    data_3[row * N + col] = tmp;

}

template<typename T,int SM,int SN,int SK> 
__global__ void mat_maul_v2(T* __restrict__ data_1,T* __restrict__ data_2,T* __restrict__ data_3,int M,int N,int K){

    int row = blockDim.y * blockIdx.y ;
    int col = blockDim.x * blockIdx.x ;
    int global_row = row + threadIdx.y;
    int global_col = col + threadIdx.x;

    if(global_row >= M || global_col >= N) return;

    
    __shared__ T left_shared[SM][SK];
    __shared__ T right_shared[SK][SN];

    T tmp = 0.0f;
    for(int step = 0;step < K;step += SK){
        if(threadIdx.y<SM && threadIdx.x < SK){
            left_shared[threadIdx.y][threadIdx.x] = data_1[(row+threadIdx.y) * K + step  + threadIdx.x];
        }
        if(threadIdx.y<SK && threadIdx.x < SN){
            right_shared[threadIdx.y][threadIdx.x] = data_2[col + threadIdx.x + (threadIdx.y + step)*N];
        }
        __syncthreads();
       
        for(int k = 0;k < SK;k++){
            tmp += left_shared[threadIdx.y][k] * right_shared[ k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row + threadIdx.y < M && col + threadIdx.x < N)

    data_3[(row + threadIdx.y) * N + (col + threadIdx.x)] = tmp;
}


template<typename T> 
__global__ void mat_maul_v3(T* __restrict__ data_1,T* __restrict__ data_2,T* __restrict__ data_3,int M,int N,int K){
    int row = blockDim.y * blockIdx.y ;
    int col = blockDim.x * blockIdx.x ;

    // 每个线程处理多个位置的元素，增加计算量；
    
}

