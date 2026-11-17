#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// check宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 静态声明共享内存
__global__ void sharedMem(
    float* input,
    float* output,
    int N
) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        smem[threadIdx.x] = input[idx];
        __syncthreads();
        int partner = blockDim.x - 1 - threadIdx.x;
        output[idx] = smem[partner];
    }
}

// 动态声明共享内存
__global__ void dynamicSharedMem(
    float* data,
    int N
) {
    extern __shared__ float smem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        smem[threadIdx.x] = data[idx];
    }
    __syncthreads();
} 

int main() {
    const int N = 256;
    const int blockSize = 256;
    const size_t bytes = N * sizeof(float);

    // 初始化主机端数据。
    float* h_data = static_cast<float*>(malloc(bytes));
    if (h_data == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // 分配设备内存，并将数据拷贝到 GPU。
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 动态共享内存大小按每个 block 的线程数计算。
    const int gridSize = (N + blockSize - 1) / blockSize;
    const size_t smemSize = blockSize * sizeof(float);
    dynamicSharedMem<<<gridSize, blockSize, smemSize>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷回结果并打印，确认 kernel 执行完成。
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    std::cout << "动态共享内存结果 (前10个): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << (i == 9 ? '\n' : ' ');
    }

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    return 0;
}
