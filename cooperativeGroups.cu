#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)



int main () {

    return 0;
}