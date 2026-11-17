#include <iostream>
#include <cuda_runtime.h>

__global__ void sharedMemDemo(float* input, float* output, int n) {
    // 静态分配共享内存 (大小为 256 个 float)
    __shared__ float smem[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 从全局内存加载到共享内存
    if (idx < n) {
        smem[threadIdx.x] = input[idx];
    }

    // 2. 同步：确保 Block 内所有线程都完成了数据加载
    __syncthreads();

    // 3. 从共享内存读取（读自己的数据和前一个线程的数据）
    if (idx < n && threadIdx.x > 0) {
        output[idx] = smem[threadIdx.x] + smem[threadIdx.x - 1];
    }
    // 处理 threadIdx.x == 0 的特殊情况，否则 output[0] 没有值
    else if (idx < n && threadIdx.x == 0) {
        output[idx] = smem[threadIdx.x];
    }
}

int main() {
    int n = 256; 
    size_t size = n * sizeof(float);

    // 1. 分配并初始化 Host 内存
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)i; // 初始化为 0, 1, 2, ..., 255
        h_output[i] = 0.0f;
    }

    // 2. 分配 Device 显存
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // 3. 将输入数据拷贝到 GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 4. 启动 Kernel
    // 注意：因为我们静态分配了 smem[256]，所以每个 Block 的线程数必须 <= 256
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sharedMemDemo<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // 5. 将结果拷贝回 Host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // 6. 打印结果验证
    std::cout << "输出结果 (前10个): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "..." << std::endl;
    // 预期输出: 
    // i=0: 0 (特殊处理)
    // i=1: 1+0 = 1
    // i=2: 2+1 = 3
    // i=3: 3+2 = 5 ...

    // 7. 清理内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}