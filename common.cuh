#include <iostream>



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