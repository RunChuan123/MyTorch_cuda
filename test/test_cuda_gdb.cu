#include <stdio.h>
#include "../utils/cuda_malloc.cuh"


__global__ void test(float* a,float* b,float* c,int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > size) return;
    c[idx] = a[idx] + b[idx];
}

int main(){
    int ts = 256;
    int bs = 3;


    float* a,*b,*c;
    cuda_malloc_batch<float>(ts*bs,&a,&b,&c);
    test<<<bs,ts>>>(a,b,c,ts*bs);
    cudaDeviceSynchronize();

    cuda_free_batch(a,b,c);



}




