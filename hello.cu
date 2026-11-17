#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    int n = 1000;
    float* d_data;

    // 1. 在 GPU 上分配显存
    cudaMalloc((void**)&d_data, n * sizeof(float));

    // 2. 调用核函数
    myKernel<<<128, 256>>>(d_data, n);

    // 3. 同步等待 GPU 报错
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel ran successfully!" << std::endl;
    }

    // 4. 释放显存
    cudaFree(d_data);
    return 0;
}