

// qkv : N * d
template<typename T,int N,int d>
__global__ void normal_atten(T* q,T* k, T* v,T* score,T* out){
    // plan:
    // a block calculate a q & all k
    __shared__ row_q[d];
    __shared__ qk_value[d];

    // load row_q
    for(int i = threadidx.x;i<d;i+=blockDim.x){
        row_q[i] = q[blockIdx.x * d + i];
    }
    for(int n=0;n<N;n++){
        T qk_mm = (T)0;
        for(int i = threadidx.x;i<d;i+=blockDim.x){
            qk_value[i] += row_q[i]*k[n*d + i];
        }
        for(int stride = d/2;i>0;i>>1){
            qk_mm
        }

    }





}