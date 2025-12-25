
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



// block(SN/x,SM/2)
// grid(N/SN,M/SM)
template<typename T,int SM,int SN,int SK> 
__global__ void mat_maul_v3(T* __restrict__ A,T* __restrict__ B,T* __restrict__ C,int M,int N,int K){
 
    const int row0 = blockIdx.y * SM;
    const int col0 = blockIdx.x * SN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int r0 = row0 + 2 * ty;
    const int c0 = col0 + 2 * tx;

    __shared__ T As[SM * SK];
    __shared__ T Bs[SN * SK];

    T c00 = (T)0,c01 = (T)0,c10 = (T)0,c11 = (T)0;

    const int tId = ty * blockDim.x + tx;
    const int nThreads = blockDim.x * blockDim.y;

    for(int k0=0;k0 < K;k0 += SK){
        // load As
        for(int idx = tId;idx < SM*SK;idx += nThreads){
            const int i = idx/SK;
            const int k = idx % SK;
            As[idx]=A[(row0+i)*K+(k0+k)];
        }
        // load Bs
        for(int idx = tId;idx < SK*SN;idx += nThreads){
            const int i = idx / SN;
            const int k = idx % SN;
            Bs[idx] = B[(k0 + i)*N + (col0 + k)];
        }
        __syncthreads();
        // load 2 * 2 index
        const int aRow0 = (2 * ty) * SK;
        const int aRow1 = (2 * ty + 1) *SK;
        const int bCol0 = (2* tx);
        const int bCol1 = (2 * tx +1);

        #pragma unroll
        for(int k=0;k < SK;k++){
            const T a0 = As[aRow0 + k];
            const T a1 = As[aRow1 + k];
            const T b0 = Bs[k * SN + bCol0];
            const T b1 = Bs[k * SN + bCol1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }
    C[r0 * N + c0] = c00;
    C[r0 * N + c0 +1] = c01;
    C[(r0 + 1) *N + c0 ] = c10;
    C[(r0 + 1) *N + c0 +1] = c11;
}


#define FLOAT4(addr) (reinterpret_cast<float4*>(addr)[0])

// block(SN/x,SM/2)
// grid(N/SN,M/SM)
template<typename T,int SM,int SN,int SK> 
__global__ void mat_maul_v4(T* __restrict__ A,T* __restrict__ B,T* __restrict__ C,int M,int N,int K){
 
    const int row0 = blockIdx.y * SM;
    const int col0 = blockIdx.x * SN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int r0 = row0 + 2 * ty;
    const int c0 = col0 + 2 * tx;

    __shared__ T As[SM * SK];
    __shared__ T Bs[SN * SK];

    T c00 = (T)0,c01 = (T)0,c10 = (T)0,c11 = (T)0;

    const int tId = ty * blockDim.x + tx;
    // SM /2 , SN /2 
    const int nThreads = blockDim.x * blockDim.y;

    for(int k0=0;k0 < K;k0 += SK){
        // load As
        const int AvecCount = (SM * SK) / 4;
        for(int v = tId;v < SM*SK;v+=nThreads){
            const int elem = v * 4;
            const int i = elem / SK;
            const int k = elem % SK;
            const float4 a4 = FLOAT4(A + (row0 + i) * SK + k0 + k);
            FLOAT4(As + i * SK + k) = a4;
        }
        // load Bs
        const int BvecCount = (SN * SK) / 4;
        for(int v = tId;v < SN*SK;v+=nThreads){
            const int elem = v * 4;
            const int i = elem / SN;
            const int k = elem % SN;
            const float4 b4 = FLOAT4(B + (k0 + i) * SK + col0 + k);
            FLOAT4(Bs + i * SN + k) = b4;
        }
        __syncthreads();
        // load 2 * 2 index
        const int aRow0 = (2 * ty) * SK;
        const int aRow1 = (2 * ty + 1) *SK;
        const int bCol0 = (2* tx);
        const int bCol1 = (2 * tx +1);

        #pragma unroll
        for(int k=0;k < SK;k++){
            const T a0 = As[aRow0 + k];
            const T a1 = As[aRow1 + k];
            const T b0 = Bs[k * SN + bCol0];
            const T b1 = Bs[k * SN + bCol1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }
    C[r0 * N + c0] = c00;
    C[r0 * N + c0 +1] = c01;
    C[(r0 + 1) *N + c0 ] = c10;
    C[(r0 + 1) *N + c0 +1] = c11;
}


