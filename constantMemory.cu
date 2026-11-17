#include <iostream> 
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int FILTER_SIZE = 256;
__constant__ float coefficients[FILTER_SIZE];

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error: %s\n",                 \
                     cudaGetErrorString(err));                    \
        return EXIT_FAILURE;                                       \
    }                                                              \
} while (0)

__global__ void applyFilter(const float* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float result = 0.0f;
        for (int i = 0; i < FILTER_SIZE; i++) {
            result += data[idx + i] * coefficients[i];
        }
        output[idx] = result;
    }
}

int main() {
    constexpr int n = 1024;
    constexpr int blockSize = 256;
    // 计算内存
    const size_t inputBytes = (n + FILTER_SIZE - 1) * sizeof(float);
    const size_t outputBytes = n * sizeof(float);

    // 主机端设置滤波值
    float h_coeff[FILTER_SIZE];
    for (float& value : h_coeff) value = 1.0f / FILTER_SIZE;
    // 分配主机端内存
    float* h_data = static_cast<float*>(std::malloc(inputBytes));
    float* h_output = static_cast<float*>(std::malloc(outputBytes));
    if (!h_data || !h_output) return EXIT_FAILURE;
    // 主机端输入初始化
    for (int i = 0; i < n + FILTER_SIZE - 1; ++i) h_data[i] = 1.0f;

    // 分配设备显存
    float *d_data = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    // host -> device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, inputBytes, cudaMemcpyHostToDevice));
    __syncthreads();
    // 把host的滤波器传给gpu的常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(coefficients, h_coeff, sizeof(h_coeff)));

    int gridSize = (n + blockSize - 1) / blockSize;
    // 调用kernel函数
    applyFilter<<<gridSize, blockSize>>>(d_data, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    // device -> host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < n; i++) {
    //     std::cout << h_output[i];
    // }
    cudaFree(d_data);
    cudaFree(d_output);
    std::free(h_data);
    std::free(h_output);
    return 0;
}
