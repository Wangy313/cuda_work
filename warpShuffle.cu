#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define WARP_SIZE 32

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/*
不需要共享内存，访问同一warp内的所有线程的值，直接访问寄存器
T __shfl_sync(unsigned mask, T var, int srcLane, int width=32); 从执行lane取
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width=32); 从低lane取
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=32); 从高lane取
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=32); 与xor配对的lane取
mask参与的线程的掩码 通常用 0xFFFFFFFF 表示所有 32 个线程参与
var当前线程要分享的值
width逻辑子warp的宽度（两个线程之间的间距）
*/

// =====================================================
// 1. Warp Broadcast：广播操作
// 所有线程获取指定 Lane 的数据
// =====================================================
__global__ void warp_broadcast(
    const float* input,
    float* output,
    int srcLane
) {
    int tid = threadIdx.x;

    float val = input[tid];

    float result = __shfl_sync(0xFFFFFFFF, val, srcLane);

    output[tid] = result;
}


// =====================================================
// 2. Warp Reduce：Warp 内求和归约
// 只有 Lane 0 保存最终正确结果
// =====================================================
__global__ void warp_reduce_sum(
    const float* input,
    float* output
) {
    int tid = threadIdx.x;

    float val = input[tid];

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xFFFFFFFF,
            val,
            offset
        );
    }

    // 只有 Lane 0 保存完整的求和结果
    if (tid == 0) {
        output[0] = val;
    }
}

// =====================================================
// CPU 主函数
// =====================================================
int main() {

    // 1. 设置输入规模
    const int N = WARP_SIZE;

    const int blockSize = WARP_SIZE;
    const int gridSize = 1;

    size_t bytes = (size_t)N * sizeof(float);

    printf("Input elements: %d\n", N);
    printf("Block size: %d\n", blockSize);
    printf("Grid size: %d\n", gridSize);


    // =================================================
    // 2. CPU 端分配内存
    // =================================================

    float* h_input = (float*)malloc(bytes);
    float* h_broadcast = (float*)malloc(bytes);

    if (h_input == NULL || h_broadcast == NULL) {
        fprintf(stderr, "Host memory allocation failed!\n");
        free(h_input);
        free(h_broadcast);
        return EXIT_FAILURE;
    }

    // 初始化输入数据：1, 2, 3, ..., 32
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i + 1);
    }


    // =================================================
    // 3. GPU 端分配内存
    // =================================================

    float* d_input;
    float* d_broadcast;
    float* d_sum;

    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_broadcast, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(float)));


    // =================================================
    // 4. 将输入数据从 CPU 复制到 GPU
    // =================================================

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice
    ));


    // =================================================
    // 5. 启动 GPU Kernel
    // =================================================

    // 5.1 Warp Broadcast
    // 让所有线程获取 Lane 5 的值
    int srcLane = 5;

    warp_broadcast<<<gridSize, blockSize>>>(
        d_input,
        d_broadcast,
        srcLane
    );

    CUDA_CHECK(cudaGetLastError());


    // 5.2 Warp Reduce Sum
    // 计算 1 + 2 + ... + 32
    warp_reduce_sum<<<gridSize, blockSize>>>(
        d_input,
        d_sum
    );

    CUDA_CHECK(cudaGetLastError());

    // 等待 GPU 完成计算
    CUDA_CHECK(cudaDeviceSynchronize());


    // =================================================
    // 6. 将 GPU 计算结果复制回 CPU
    // =================================================

    CUDA_CHECK(cudaMemcpy(
        h_broadcast,
        d_broadcast,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    float h_sum;

    CUDA_CHECK(cudaMemcpy(
        &h_sum,
        d_sum,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // =================================================
    // 7. 输出 Warp Broadcast 结果
    // =================================================

    printf("\n========== Warp Broadcast ==========\n");

    printf("Source Lane: %d\n", srcLane);

    for (int i = 0; i < N; i++) {
        printf("Lane %2d: %.0f\n", i, h_broadcast[i]);
    }


    // =================================================
    // 8. 输出 Warp Reduce 结果
    // =================================================

    float expected = 0.0f;

    for (int i = 0; i < N; i++) {
        expected += h_input[i];
    }

    printf("\n========== Warp Reduce Sum ==========\n");

    printf("GPU result: %.0f\n", h_sum);
    printf("Expected:   %.0f\n", expected);

    if (h_sum == expected) {
        printf("Verification: PASS\n");
    } else {
        printf("Verification: FAIL\n");
    }


    // =================================================
    // 9. 释放 GPU 和 CPU 内存
    // =================================================

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_broadcast));
    CUDA_CHECK(cudaFree(d_sum));

    free(h_input);
    free(h_broadcast);

    return 0;
}