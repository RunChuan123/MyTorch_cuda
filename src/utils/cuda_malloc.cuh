#include <type_traits>
#include <stdio.h>
#include <stdlib.h>



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




template<typename T, typename... Args>
void cuda_malloc_batch(int size, Args... pointers) {
    (cudaMalloc(pointers, size * sizeof(T)), ...);
}

template<typename... Args>
void cuda_free_batch(Args... pointers) {
    (cudaFree(pointers), ...);
}

