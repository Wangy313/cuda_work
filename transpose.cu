#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>

// =====================================================
// CUDA 错误检查宏
// =====================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_DIM 32
#define BLOCK_ROWS 8


// =====================================================
// GPU Kernel：共享内存优化矩阵转置
// =====================================================
__global__ void transpose_shared(
    const float* input,
    float* output,
    int width,
    int height
) {

    // 1. 定义共享内存
    // 多增加一列，消除 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // =================================================
    // 2. 计算当前线程对应的输入矩阵坐标
    // =================================================

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;


    // =================================================
    // 3. Global Memory → Shared Memory
    //    合并读取，写入共享内存
    // =================================================

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {

        if (x < width && y + j < height) {

            // 从全局内存连续读取
            float val = input[(y + j) * width + x];

            // 写入共享内存
            tile[threadIdx.y + j][threadIdx.x] = val;   

        } else {

            tile[threadIdx.y + j][threadIdx.x] = 0.0f;
        }
    }

    // 等待所有线程完成共享内存写入
    __syncthreads();


    // =================================================
    // 4. 交换 Block 的行列位置
    //    为转置后的矩阵计算坐标
    // =================================================

    int outX = blockIdx.y * TILE_DIM + threadIdx.x;
    int outY = blockIdx.x * TILE_DIM + threadIdx.y;


    // =================================================
    // 5. Shared Memory → Global Memory
    //    从共享内存按列读取，合并写入全局内存
    // =================================================

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {

        if (outX < height && outY + j < width) {

            // 从共享内存按列读取
            float val = tile[threadIdx.x][threadIdx.y + j];

            // 连续写入转置后的矩阵
            output[(outY + j) * height + outX] = val;
        }
    }
}


// =====================================================
// CPU 主函数
// =====================================================
int main() {

    // 1. 设置矩阵大小
    const int width = 1003;
    const int height = 997;

    // 每个 Block：32 × 8 = 256 个线程
    dim3 block(TILE_DIM, BLOCK_ROWS);

    // 计算 Grid 大小
    dim3 grid(
        (width + TILE_DIM - 1) / TILE_DIM,
        (height + TILE_DIM - 1) / TILE_DIM
    );

    // 计算内存大小
    size_t bytes = (size_t)width * height * sizeof(float);

    printf("Matrix size: %d x %d\n", height, width);
    printf("Block size: %d x %d\n", TILE_DIM, BLOCK_ROWS);
    printf("Grid size: %d x %d\n", grid.x, grid.y);


    // =================================================
    // 2. CPU 端分配内存
    // =================================================

    std::vector<float> h_input(width * height);
    std::vector<float> h_output(width * height);

    // 初始化输入矩阵
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            h_input[row * width + col] = row * 10000 + col;
        }
    }


    // =================================================
    // 3. GPU 端分配内存
    // =================================================

    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));


    // =================================================
    // 4. CPU → GPU
    // =================================================

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));


    // =================================================
    // 5. 启动 GPU Kernel
    // =================================================

    transpose_shared<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // =================================================
    // 6. GPU → CPU
    // =================================================

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));


    // =================================================
    // 7. CPU 验证转置结果
    // =================================================

    bool pass = true;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            // 原矩阵 A[row][col]
            float expected = h_input[row * width + col];

            // 转置矩阵 B[col][row]
            float actual = h_output[col * height + row];

            if (actual != expected) {

                printf("Error at (%d, %d)\n", row, col);
                printf("Expected: %.0f\n", expected);
                printf("Actual: %.0f\n", actual);

                pass = false;
                break;
            }
        }

        if (!pass) break;
    }


    // =================================================
    // 8. 输出结果
    // =================================================

    printf("\n========== Result ==========\n");

    if (pass) {
        printf("Transpose: PASS\n");
    } else {
        printf("Transpose: FAIL\n");
    }


    // =================================================
    // 9. 释放内存
    // =================================================

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return pass ? 0 : 1;
}