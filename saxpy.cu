#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int N = 128;

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void saxpy(
    float k,
    float* X,
    float* Y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = k * X[idx] + Y[idx];
    }
}

int main () {
    // 计算线程
    const int blockSize = 32;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 计算内存大小
    size_t bytes = N * sizeof(float);
    // cpu端分配内存
    float k = 3.0f;
    float* x_h = (float*)malloc(bytes);
    float* y_h = (float*)malloc(bytes);
    // 初始化
    for (int i = 0; i < N; i++) {
        x_h[i] = 1.0f;
        y_h[i] = 0.0f;
    }
    // gpu分配显存
    float *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&x_d, bytes));
    CUDA_CHECK(cudaMalloc(&y_d, bytes));
    
    // host to device
    CUDA_CHECK(cudaMemcpy(x_d, x_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y_h, bytes, cudaMemcpyHostToDevice));
    
    // 启动kernel
    saxpy<<<gridSize, blockSize>>>(k, x_d, y_d);

    // device to host
    CUDA_CHECK(cudaMemcpy(x_h, x_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_h, y_d, bytes, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; i++) {
        std::cout << y_h[i] << " ";
    }

    free(x_h);
    free(y_h);
    cudaFree(x_d);
    cudaFree(y_d);
    
    return 0;
}
