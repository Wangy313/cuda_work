#include <cstdio>       // printf、fprintf
#include <cstdlib>      // malloc、free、exit
#include <cmath>        // sinf、cosf、fabsf
#include <chrono>       // CPU 计时
#include <cuda_runtime.h> // CUDA Runtime API

// 错误检查宏
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// GPU Kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 一个线程负责计算一个数组元素。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU 基线
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    // CPU 逐元素计算，用于性能对比和结果校验。
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 24;  // 向量元素个数，约 1600 万
    const size_t size = N * sizeof(float);
    printf("Vector size: %d elements (%.1f MB)\n", N, size / (1024.0 * 1024.0));

    // ========== 分配主机内存 ==========
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);

    // 初始化输入向量，构造一组可重复的测试数据。
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i * 0.001f);
        h_B[i] = cosf(i * 0.001f);
    }

    // ========== CPU 计算：作为基准结果 ==========
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("CPU time: %.4f ms\n", cpu_ms);

    // ========== GPU 计算 ==========
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 创建 CUDA Event，用于测量 GPU 端耗时。
    cudaEvent_t start, stop, kernel_start, kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // 端到端计时：包括主机与设备之间的数据传输。
    cudaEventRecord(start);

    // 将输入向量从 Host 拷贝到 Device。
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int blockSize = 256;                    // 每个 block 的线程数
    int gridSize = (N + blockSize - 1) / blockSize; // 覆盖全部元素

    // 仅测量 Kernel 执行时间，不包含内存拷贝。
    cudaEventRecord(kernel_start);
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(kernel_stop);

    CUDA_CHECK(cudaGetLastError());
    // 将 GPU 计算结果拷回 Host。
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_total_ms = 0, kernel_ms = 0;
    cudaEventElapsedTime(&gpu_total_ms, start, stop);
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

    printf("GPU kernel time: %.4f ms\n", kernel_ms);
    printf("GPU total time (with memcpy): %.4f ms\n", gpu_total_ms);
    printf("Speedup (kernel only): %.1fx\n", cpu_ms / kernel_ms);
    printf("Speedup (end-to-end): %.1fx\n", cpu_ms / gpu_total_ms);

    // ========== 结果验证：比较 CPU 和 GPU 输出 ==========
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-5f) {
            if (errors < 5) {
                printf("Mismatch at %d: CPU=%.6f, GPU=%.6f\n",
                       i, h_C_cpu[i], h_C_gpu[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED: %d mismatches\n", errors);
    }

    // ========== 计算有效带宽 ==========
    // 向量加法读 2 个 float、写 1 个 float = 12 bytes/element
    float bandwidth_gb = (3.0f * size) / (kernel_ms * 1e-3f) / 1e9f;
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gb);

    // ========== 清理 ==========
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
