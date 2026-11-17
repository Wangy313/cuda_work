#include <iostream>
#include <cuda_runtime.h>

__global__ void Kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { // 严格对齐 n
        float a = data[idx];
        float b = a * 2.0f;
        float result = a + b;
        data[idx] = result;
    }
}

int main() {
    int n = 256; 
    size_t size = n * sizeof(float);

    float* h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = (float)i;

    float* d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // 启动配置：256 个线程刚好一个 Block
    Kernel<<<1, 256>>>(d_data, n);

    // 强制检查 CUDA 错误！
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    std::cout << "计算结果: ";
    for (int i = 0; i < 10; i++) std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);
    free(h_data);
    return 0;
}