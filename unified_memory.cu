#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vadd(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;

    size_t bytes = (size_t)N * sizeof(float);

    // 分配统一内存
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    int blockSize = 256;
    int gridSize = (blockSize + N - 1) / blockSize;

    vadd<<<gridSize, blockSize>>>(a, b, c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool pass = true;

    for (int i = 0; i < N; i++) {
        if (c[i] != 3.0f) {
            printf("Error at index %d: %f\n", i, c[i]);
            pass = false;
            break;
        }
    }

    if (pass) {
        printf("Verification: PASS\n");
    } else {
        printf("Verification: FAIL\n");
    }

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    return 0;
}