#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 向量加法的 Kernel 函数
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 1. 设置数据规模
    const int N = 1 << 20; // 100万个元素
    const int numStreams = 2; // 使用 2 个流来实现流水线重叠
    const int chunkSize = N / numStreams; // 每个流处理 50万个元素
    
    size_t bytes = N * sizeof(float);
    size_t chunkBytes = chunkSize * sizeof(float);

    // ========================================================
    // 核心准备：分配【锁页内存】（Pinned Memory）
    // 只有锁页内存才能进行真正的异步传输！
    // ========================================================
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, bytes));

    // 初始化 CPU 数据
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配 GPU 显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ========================================================
    // 创建 2 个 Stream，它们是异步流水线的轨道
    // ========================================================
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 计算 Kernel 启动参数
    int blockSize = 256;
    int gridSize = (chunkSize + blockSize - 1) / blockSize;

    // ========================================================
    // 核心逻辑：异步提交任务
    // ========================================================
    for (int i = 0; i < numStreams; i++) {
        // 计算当前流处理的数据偏移量
        int offset = i * chunkSize;
        
        // 1. 异步拷贝（Host -> Device）
        // 注意：这里加上了 Stream 参数，CPU 发完指令立刻走，不阻塞
        CUDA_CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));

        // 2. 异步执行计算
        // 在 Kernel 启动的 <<<>>> 中，第 4 个参数（0的位置）就是 Stream 指定
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, chunkSize);

        // 3. 异步拷贝（Device -> Host）
        // 计算完的数据，交给该 Stream 异步搬回 CPU
        CUDA_CHECK(cudaMemcpyAsync(h_C + offset, d_C + offset, chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // ========================================================
    // 同步：CPU 在这里死等，直到所有 Stream 里的任务全部完成
    // ========================================================
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // 验证结果（前 10 个）
    std::cout << "验证结果 (前 10 个): ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "..." << std::endl;

    // ========================================================
    // 清理现场（一定要释放锁页内存和 Stream）
    // ========================================================
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}