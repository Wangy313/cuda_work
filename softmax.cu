#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        std::cerr << "CUDA Error: "                     \
                  << cudaGetErrorString(err) << "\n";   \
        std::exit(EXIT_FAILURE);                       \
    }                                                  \
} while (0)

// softmax_v0
// M行、N列 - 矩阵形状(M, N) = (4096, 4096)
__global__ void softmax_v0(float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }
    // 地址映射
    float* x = input + row * N;
    float* y = output + row * N;

    // Pass 1: 计算最大值
    float max_val = -INFINITY;
    for (int j = 0; j < N; j++) {
        max_val = fmaxf(max_val, x[j]);
    }
    // Pass 2: 计算指数和
    float sum_exp = 0.0f;
    for (int j = 0; j < N; j++) {
        sum_exp += expf(x[j] - max_val);
    }
    // Pass 3: 计算 softmax
    float inv_sum_exp = 1.0f / sum_exp;
    for (int j = 0; j < N; j++) {
        y[j] = expf(x[j] - max_val) * inv_sum_exp;
    }
}

// ------------------------------------------------分隔符------------------------------------------------

// softmax_v1 - 一个 Block 处理一行，Block 内并行规约
/*
    调用方式：
    int block_size = 256;
    int smem_size = block_size * sizeof(float);
    softmax_v1<<<M, block_size, smem_size>>>(d_input, d_output, M, N);
*/
__global__ void softmax_v1(float* input, float* output, int M, int N) {
    extern __shared__ float smem[];// 动态共享内存
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* x = input + row * N;
    float* y = output + row * N;
    // Pass 1: 计算最大值 - 每个线程处理一部分数据，整合到一个block内部
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();
    // Block 内规约求最大值 - 最大值放入 smem[0]
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] = fmaxf(smem[tid], smem[tid + step]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();
    // Pass 2: 计算指数和 - 每个线程处理一部分数据，整合到一个block内部
    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum_exp += expf(x[i] - max_val);
    }
    smem[tid] = sum_exp;
    __syncthreads();
    // Block 内规约求和 - 和放入 smem[0]
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    sum_exp = smem[0];
    __syncthreads();
    // Pass 3: 计算 softmax - 每个线程处理一部分数据
    float inv_sum_exp = 1.0f / sum_exp;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum_exp;
    }
}

// ------------------------------------------------分隔符------------------------------------------------

// softmax_v2 - Warp Shuffle 优化规约
/*
    Warp 内 Shuffle + Warp 间 Shared Memory的两级结构：
    先在每个 Warp 内用 Shuffle 规约出局部结果，
    再把每个 Warp 的结果写到一个 32 元素的 SMEM 数组，
    由第一个 Warp 做最终规约。
*/

// Warp 内规约：Max
__device__ __forceinline__ float warpReduceMax(
    float val
) {
    for (int step = 16; step > 0; step >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, step));
    }
    return val;
    // 每个warp的lane0是局部最大值
}

// Block 内Max
__device__ __forceinline__ float blockReduceMax(
    float val
) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    val = warpReduceMax(val);

    // 把每个warp的第一个线程也就是lane0，放入共享内存，做共享内存规约
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    // 对smem做warp规约
    int warp_num = blockDim.x / 32;
    val = (lane < warp_num) ? smem[lane] : -INFINITY;
    if (wid == 0) {// 第一个warp
        val = warpReduceMax(val);
    }

    // 第一个warp的第一个线程就是最终值，广播给所有的线程
    __shared__ float block_result;
    if (tid == 0) {
        block_result = val;
    }
    __syncthreads();
    return block_result;
}

// Warp 内规约：Sum
__device__ __forceinline__ float warpReduceSum(
    float val
) {
    for (int step = 16; step > 0; step >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, step);
    }
    return val;
}

// Block 内Sum
__device__  __forceinline__ float blockReduceSum(
    float val
) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    val = warpReduceSum(val);

    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    int warp_num = blockDim.x / 32;
    val = (lane < warp_num) ? smem[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    __shared__ float block_sum;
    if (tid == 0) {
        block_sum = val;
    }
    __syncthreads();
    return block_sum;
}

// softmax_v2 - 一个block处理一行
__global__ void softmax_v2(
    float* input,
    float* output,
    int M,
    int N
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    // 取得行映射
    float* x = input + row * N;
    float* y = output + row * N;

    // Pass 1: 求行最大值
    float local_max = -INFINITY;// 先把每个线程的最大值找出来
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMax(local_max);

    // Pass 2: 指数求和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSum(local_sum);

    // Pass 3: 归一化
    float inv_sum = 1.0 / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// softmax_v3 - 向量化加载 float4
__global__ void softmax_v3(
    float* input,
    float* output,
    int M, int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + N * row;
    float* y = output + N * row;

    // 向量化指针 - 按照4位一个位置来取
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    // 元素数量
    int N4 = N / 4;

    // Pass 1: 向量化求最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)));
    }
    // 处理尾部元素（N不是4的倍数）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMax(local_max);

    // Pass 2: 向量化指数求和
    float local_sum = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_sum += expf(data.x - max_val) + expf(data.y - max_val) + expf(data.z - max_val) + expf(data.w - max_val);
    }
    // 处理尾部元素（N不是4的倍数）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum_val = blockReduceSum(local_sum);

    // Pass 3: 向量化归一化
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 res;
        res.x = expf(data.x - max_val) * inv_sum;
        res.y = expf(data.y - max_val) * inv_sum;
        res.z = expf(data.z - max_val) * inv_sum;
        res.w = expf(data.w - max_val) * inv_sum;
        y4[i] = res;
    }
    // 处理尾部元素（N不是4的倍数）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// ------------------------------------------------分隔符------------------------------------------------
// softmax_v4 - 两遍融合 Kernel
__global__ void softmax_v4(
    float* input, 
    float* output,
    int M, 
    int N
) {
    extern __shared__ float smem[];// 大小 = N（暂存 exp 值）
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    int N4 = N / 4;

    // Pass 1: 求行最大值（向量化 + warp shuffle）
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)));
    }
    for (int i = 4 * N4 + tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMax(local_max);

    // Pass 2: 计算 exp 并暂存，同时累加求 sum
    // 因为这里要暂存到共享内存，共享内存只能存标量
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        smem[i] = exp_val;// 暂存到共享内存
        local_sum += exp_val;
    }
    float sum_val = blockReduceSum(local_sum);

    // 直接从 Shared Memory 读取 exp 值并归一化写出
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = smem[i] * inv_sum;
    }
}

int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int BLOCK_SIZE = 256;
    constexpr int REPEATS = 20;
    const size_t elements = static_cast<size_t>(M) * N;
    const size_t bytes = elements * sizeof(float);

    std::vector<float> h_input(elements), h_reference(elements);
    for (size_t i = 0; i < elements; ++i) {
        h_input[i] = 4.0f * std::sin(static_cast<float>(i % 997) * 0.013f);
    }

    // CPU reference，用于检查每个 GPU 版本的最大绝对误差。
    for (int row = 0; row < M; ++row) {
        const float* x = h_input.data() + static_cast<size_t>(row) * N;
        float* y = h_reference.data() + static_cast<size_t>(row) * N;
        float max_value = -INFINITY;
        for (int col = 0; col < N; ++col) max_value = std::max(max_value, x[col]);
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) sum += std::exp(x[col] - max_value);
        for (int col = 0; col < N; ++col) y[col] = std::exp(x[col] - max_value) / sum;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // memoryClockRate 的单位是 kHz；显存为 DDR，因此乘以 2，结果换算为 GB/s。
    const double peak_bandwidth_gbps = static_cast<double>(prop.memoryClockRate) * 1000.0 *
                                       (static_cast<double>(prop.memoryBusWidth) / 8.0) * 2.0 / 1.0e9;

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    struct Result { std::string name; float time_us; double bandwidth_gbps; double utilization; double speedup; float error; };
    std::vector<Result> results;

    auto run = [&](const char* name, size_t shared_bytes, auto launch) {
        CUDA_CHECK(cudaMemset(d_output, 0, bytes));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < REPEATS; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_input.data(), d_output, bytes, cudaMemcpyDeviceToHost));
        float max_error = 0.0f;
        for (size_t i = 0; i < elements; ++i) max_error = std::max(max_error, std::fabs(h_input[i] - h_reference[i]));
        const double time_us = elapsed_ms * 1000.0 / REPEATS;
        const double bandwidth_gbps = static_cast<double>(2.0 * bytes) / (time_us * 1000.0);
        const double utilization = bandwidth_gbps / peak_bandwidth_gbps * 100.0;
        results.push_back({name, static_cast<float>(time_us), bandwidth_gbps, utilization, 0.0, max_error});
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };

    const dim3 grid(M), block(BLOCK_SIZE);
    run("softmax_v0", 0, [&] { softmax_v0<<<grid, block>>>(d_input, d_output, M, N); });
    run("softmax_v1", BLOCK_SIZE * sizeof(float), [&] { softmax_v1<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, M, N); });
    run("softmax_v2", 0, [&] { softmax_v2<<<grid, block>>>(d_input, d_output, M, N); });
    run("softmax_v3", 0, [&] { softmax_v3<<<grid, block>>>(d_input, d_output, M, N); });
    run("softmax_v4", N * sizeof(float), [&] { softmax_v4<<<grid, block, N * sizeof(float)>>>(d_input, d_output, M, N); });

    const float baseline_time = results.front().time_us;
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Shape: M=" << M << ", N=" << N << ", repeats=" << REPEATS << "\n\n";
    std::cout << std::left << std::setw(14) << "Version"
              << std::right << std::setw(14) << "Time(us)"
              << std::setw(18) << "Bandwidth(GB/s)"
              << std::setw(18) << "Utilization(%)"
              << std::setw(16) << "RelSpeedup" << "\n";
    for (auto& r : results) {
        r.speedup = static_cast<double>(baseline_time) / r.time_us;
        std::cout << std::left << std::setw(14) << r.name
                  << std::right << std::setw(14) << std::fixed << std::setprecision(3) << r.time_us
                  << std::setw(18) << std::setprecision(2) << r.bandwidth_gbps
                  << std::setw(18) << r.utilization
                  << std::setw(16) << r.speedup << "\n";
        std::cout << "  max abs error = " << std::scientific << r.error
                  << (r.error < 1e-5f ? " [PASS]\n" : " [FAIL]\n");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
