#include "/root/autodl-tmp/MiniTorch/src/ops_ori/gemm.cu"
#include <stdio.h>
#include <iostream>
#define CEIL_DIV(M,N) ((M+N-1)/N)

void init(float* data,int m,int n){
    for(int i = 0;i<m;i++){
        for(int j = 0;j < n;j++){
            *(data+i*n+j) = i<<2 * j <<1 + i%3 + j % 9 + i-j -9;
        }
    }
}

void cpu_multi(float* data1,float* data2,float* data3,int m,int n,int k){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            for(int l=0;l<k;l++){
                *(data3+i*n+j) += *(data1+i*k+l) * *(data2 + l * n + j);
            }
        }
    }
}

bool checksame(float* a, float* b,int m,int n){

    for(int i=0;i<m;i++){
        for(int j=0;j< n;j++){
            if(*(a+i*n+j) != *(b+i*n+j))return false;
        }
    }
    return true;
}

int main(){

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p,0);
    std::cout << p.sharedMemPerBlock;

    int M = 1024;
    int N = 1024;
    int K = 1024;

    float* h_data_1 = (float*)malloc(sizeof(float) * M * K);
    float* h_data_2 = (float*)malloc(sizeof(float) * N * K);
    float* h_data_3 = (float*)malloc(sizeof(float) * M * N);
    memset(h_data_3,0,sizeof(float) * M * N);
    float* h_data_4 = (float*)malloc(sizeof(float) * M * N);
    float* d_data_1,*d_data_2,*d_data_3;
    init(h_data_1,M,K);
    init(h_data_2,K,N);
    cpu_multi(h_data_1,h_data_2,h_data_3,M,N,K);

    cudaStream_t s0;
    cudaStreamCreate(&s0);

    cudaMallocAsync(&d_data_1,sizeof(float) * M * K,s0);
    cudaMallocAsync(&d_data_2,sizeof(float) * N * K,s0);
    cudaMallocAsync(&d_data_3,sizeof(float) * M * N,s0);

    cudaMemsetAsync(d_data_1,0,sizeof(float) * M * K,s0);
    cudaMemsetAsync(d_data_2,0,sizeof(float) * N * K,s0);
    cudaMemsetAsync(d_data_3,0,sizeof(float) * M * N,s0);
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);



    cudaMemcpyAsync(d_data_1,h_data_1,sizeof(float) * M * K,cudaMemcpyHostToDevice,s0);
    cudaMemcpyAsync(d_data_2,h_data_2,sizeof(float) * N * K,cudaMemcpyHostToDevice,s0);
 
 
    
    cudaEventRecord(start,s0);
    constexpr int SM = 16;
    constexpr int SN = 16;
    
    constexpr int SK = 16;
    dim3 block(8,8);
    dim3 grid(CEIL_DIV(N,SN),CEIL_DIV(M,SM));
    mat_maul_v4<float,SM,SN,SK><<<grid,block,0,s0>>>(d_data_1,d_data_2,d_data_3,M,N,K);
    cudaEventRecord(end,s0);
    cudaEventSynchronize(end);


    cudaMemcpyAsync(h_data_4,d_data_3,sizeof(float) * M * N,cudaMemcpyDeviceToHost,s0);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);   
    std::cout <<std::endl << "elapsed time:" << milliseconds << std::endl;
    if(checksame(h_data_3,h_data_4,M,N)){
        std::cout << "right";
    }
    else{
        std::cout << "false";
    }

    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaFree(d_data_3);
    free(h_data_1);
    free(h_data_2);
    free(h_data_3);
    free(h_data_4);
    

}
