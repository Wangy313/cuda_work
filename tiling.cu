#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int TILE = 16;

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void matmulTiled(float* A, float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 协作加载 tile 到共享内存
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 在共享内存上计算
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


int main () {
    const int M = 64, N = 64, K = 64;
    const size_t aBytes = M * K * sizeof(float);
    const size_t bBytes = K * N * sizeof(float);
    const size_t cBytes = M * N * sizeof(float);

    float* h_A = static_cast<float*>(malloc(aBytes));
    float* h_B = static_cast<float*>(malloc(bBytes));
    float* h_C = static_cast<float*>(malloc(cBytes));
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, aBytes));
    CUDA_CHECK(cudaMalloc(&d_B, bBytes));
    CUDA_CHECK(cudaMalloc(&d_C, cBytes));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, aBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bBytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, cBytes, cudaMemcpyDeviceToHost));

    std::cout << "C[0][0] = " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
