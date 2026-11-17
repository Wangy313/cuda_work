#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <vector>

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        std::cerr << "CUDA Error: "                     \
                  << cudaGetErrorString(err) << "\n";   \
        std::exit(EXIT_FAILURE);                       \
    }                                                  \
} while (0)

// online_softmax_v0 - 单线程 Online Softmax（一个线程处理一行）
// 串行地维护 (m,d) 状态，最后再做一遍归一化。
__global__ void online_softmax_v0(
    float* input,
    float* output,
    int M,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) {
        return;
    }
    float* x = input + row * N;
    float* y = output + row * N;
    
    // 第 1 遍：Online 同时计算 max 和 sum
    float mx = -INFINITY;
    float d = 0.0f;
    for (int i = 0; i < N; i++) {
        float xi = x[i];
        float mx_new = fmaxf(mx, xi);
        d = d * expf(mx - mx_new) + expf(xi - mx_new);
        mx = mx_new;
    }
    // 第 2 遍：归一化
    float inv_val = 1.0f / d;
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - mx) * inv_val;
    }
}

// ------------------------------------------------分隔符------------------------------------------------

// online_softmax_v1 - Block 级并行 + Warp Shuffle;
// 合并规约 Warp 内 Shuffle + Warp 间 Shared Memory
__device__ __forceinline__ void warpReduceOnline(
    float& m,
    float& d
) {
    // 这里可以理解为(m, d)成对出现的
    for (int step = 16; step > 0; step >>= 1) {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, step);
        float d2 = __shfl_down_sync(0xFFFFFFFF, d, step);
        float m_new = fmaxf(m, m2);
        // 空状态的权重为零，避免 -INFINITY - (-INFINITY) 产生 NaN。
        d = (d > 0.0f ? d * expf(m - m_new) : 0.0f) + (d2 > 0.0f ? d2 * expf(m2 - m_new) : 0.0f);
        m = m_new;
    }
} 

__global__ void online_softmax_v1(
    float* input,
    float* output,
    int M, 
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input + row * N;
    float* y = output + row * N;
    
    // 1.每线程online处理一段
    float local_m = -INFINITY;
    float local_d = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(xi, local_m);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 第一级：warp内规约
    warpReduceOnline(local_m, local_d);

    // Warp间通过共享内存转换
    __shared__ float m_smem[32];
    __shared__ float d_smem[32];

    // warp内第一个线程
    if (lane == 0) {
        m_smem[wid] = local_m;
        d_smem[wid] = local_d;
    }
    __syncthreads();

    // 第二级：warp间规约
    int warp_num = blockDim.x / 32;
    if (wid == 0) {
        local_m = lane < warp_num ? m_smem[lane] : -INFINITY;
        local_d = lane < warp_num ? d_smem[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    // 广播结果
    __shared__ float m_res, d_res;
    if (tid == 0) {
        m_res = local_m;
        d_res = local_d;
    }
    __syncthreads();

    // 归一化
    float inv_val = 1.0f / d_res;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - m_res) * inv_val;
    }
}

// ------------------------------------------------分隔符------------------------------------------------

// online_softmax_v2 - 向量化加载 float4
__global__ void online_softmax_v2(
    float* input,
    float* output,
    int M,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input + row * N;
    float* y = output + row * N;

    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // 第一遍：向量化online递推
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        // 对4个元素依次更新(m, d)
        float val[4] = {data.x, data.y, data.z, data.w};
        for (int j = 0; j < 4; j++) {
            float m_new = fmaxf(val[j], local_m);
            local_d = local_d * expf(local_m - m_new) + expf(val[j] - m_new);
            local_m = m_new;
        }
    }
    // 尾部处理
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(xi, local_m);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // warp内规约
    warpReduceOnline(local_m, local_d);

    // warp间规约
    __shared__ float d_mem[32];
    __shared__ float m_mem[32];
    if (lane == 0) {
        d_mem[wid] = local_d;
        m_mem[wid] = local_m;
    }
    __syncthreads();
    int warp_num = blockDim.x / 32;
    if (wid == 0) {
        local_m = lane < warp_num ? m_mem[lane] : -INFINITY;
        local_d = lane < warp_num ? d_mem[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    __shared__ float m_res, d_res;
    if (tid == 0) {
        m_res = local_m;
        d_res = local_d;
    }
    __syncthreads();

    // 归一化
    float inv_val = 1.0f / d_res;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 res;
        res.x = expf(data.x - m_res) * inv_val;
        res.y = expf(data.y - m_res) * inv_val;
        res.z = expf(data.z - m_res) * inv_val;
        res.w = expf(data.w - m_res) * inv_val;
        y4[i] = res;
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - m_res) * inv_val;
    }
}

// ------------------------------------------------分隔符------------------------------------------------

// online_softmax_v3 - 寄存器缓存消除第二遍读取
// 寄存器缓存实现 One-Pass Softmax
// 适用于 N / blockDim.x 较小的场景（每线程处理元素数不超过 MAX_ELEMS）
// N/blockDim≤ 几十
// V3 适合 N≤8192 且 blockDim=256 的场景。对于更大的 N，应退回 V2（两遍方案）。
#define MAX_ELEMS_PER_THREAD 32

__global__ void online_softmax_v3(
    float* input,
    float* output,
    int M,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    float* x = input + row * N;
    float* y = output + row * N;

    // 寄存器数组缓存输入数据
    float reg_cache[MAX_ELEMS_PER_THREAD];

    // 第一遍 （唯一一次）：读数据+online递推
    float local_m = -INFINITY;
    float local_d = 0.0f;
    int cnt = 0;

    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        reg_cache[cnt] = xi;// 缓存到寄存器
        cnt++;
        float m_new = fmaxf(xi, local_m);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }
    // 两级warp shuffle合并
    warpReduceOnline(local_m, local_d);

    __shared__ float m_smem[32];
    __shared__ float d_smem[32];
    if (lane == 0) {
        m_smem[wid] = local_m;
        d_smem[wid] = local_d;
    }
    __syncthreads();

    int warp_num = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < warp_num) ? m_smem[lane] : -INFINITY;
        local_d = (lane < warp_num) ? d_smem[lane] : 0.0f;
        warpReduceOnline(local_m, local_d);
    }

    __shared__ float m_res, d_res;
    if (tid == 0) {
        m_res = local_m;
        d_res = local_d;
    }
    __syncthreads(); // 等待线程 0 广播行最大值和指数和。
    float inv_val = 1.0f / d_res;
    int idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(reg_cache[idx] - m_res) * inv_val;
        idx++;
    }
}

// ------------------------------------------------分隔符------------------------------------------------


// softmax_v4 - 一个block处理多行
// V4: Grid Stride 多行处理 + 所有优化集成
__global__ void online_softmax_v4(
    float* input,
    float* output,
    int M,
    int N
) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        float* x = input + row * N;
        float* y = output + row * N;

        // 向量化递推
        float4* x4 = reinterpret_cast<float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        int N4 = N / 4;

        float local_m = -INFINITY;
        float local_d = 0.0f;
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float val[4] = {data.x, data.y, data.z, data.w};
            for (int j = 0; j < 4; j++) {
                float m_new = fmaxf(local_m, val[j]);
                local_d = local_d * expf(local_m - m_new) + expf(val[j] - m_new);
                local_m = m_new;
            }
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            float xi = x[i];
            float m_new = fmaxf(local_m, xi);
            local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
            local_m = m_new;
        }
        // warp内规约
        warpReduceOnline(local_m, local_d);

        // warp间规约
        __shared__ float m_smem[32], d_smem[32];
        if (lane == 0) {
            m_smem[wid] = local_m;
            d_smem[wid] = local_d;
        }
        __syncthreads();

        int warp_num = blockDim.x / 32;
        if (wid == 0) {
            local_m = lane < warp_num ? m_smem[lane] : -INFINITY;
            local_d = lane < warp_num ? d_smem[lane] : 0.0f;
            warpReduceOnline(local_m, local_d);
        }

        __shared__ float m_res, d_res;
        if (tid == 0) {
            m_res = local_m;
            d_res = local_d;
        }
        __syncthreads();

        float inv_val = 1.0f / d_res;
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 data = x4[i];
            float4 res;
            res.x = expf(data.x - m_res) * inv_val;
            res.y = expf(data.y - m_res) * inv_val;
            res.z = expf(data.z - m_res) * inv_val;
            res.w = expf(data.w - m_res) * inv_val;
            y4[i] = res;
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            y[i] = expf(x[i] - m_res) * inv_val;
        }
        
        __syncthreads();  // 确保 Shared Memory 在下一行之前被重置
    }
}

int main() {
    // 1. 测试配置：float4 要求各行对齐；V3 的每线程缓存最多容纳 32 个元素。
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP = 3;
    constexpr int REPEATS = 20;
    static_assert(N % 4 == 0, "float4 requires aligned rows");
    static_assert(N <= BLOCK_SIZE * MAX_ELEMS_PER_THREAD, "V3 cache is too small");
    const size_t count = static_cast<size_t>(M) * N;
    const size_t bytes = count * sizeof(float);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    const int grid_size = std::min(M, num_sms * 4);
    // CUDA 显存时钟单位 kHz，DDR 乘 2，十进制 GB/s。
    const double peak_gbps = 2.0 * prop.memoryClockRate * 1000.0
                           * (prop.memoryBusWidth / 8.0) / 1e9;

    // 2. 构造输入，用 double 精度 CPU Softmax 生成独立参考答案。
    std::vector<float> input(count), reference(count), output(count);
    for (size_t i = 0; i < count; ++i)
        input[i] = 8.0f * std::sin(static_cast<float>(i % 997) * 0.013f);
    for (int row = 0; row < M; ++row) {
        const size_t base = static_cast<size_t>(row) * N;
        double maximum = -INFINITY, sum = 0.0;
        for (int col = 0; col < N; ++col)
            maximum = std::max(maximum, static_cast<double>(input[base + col]));
        for (int col = 0; col < N; ++col)
            sum += std::exp(static_cast<double>(input[base + col]) - maximum);
        for (int col = 0; col < N; ++col)
            reference[base + col] = std::exp(static_cast<double>(input[base + col]) - maximum) / sum;
    }

    // 3. 分配显存和计时事件；传输及 CPU 校验不计入 kernel 耗时。
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double baseline_us = 0.0;
    bool all_passed = true;

    std::cout << "GPU: " << prop.name << " | SMs: " << num_sms
              << " | Peak bandwidth: " << peak_gbps << " GB/s\n"
              << "M=" << M << ", N=" << N << ", block=" << BLOCK_SIZE
              << ", V4 grid=" << grid_size << ", repeats=" << REPEATS << "\n"
              << "Effective bandwidth = 2*M*N*sizeof(float)/time (one read + one write).\n"
              << "Utilization = effective bandwidth / theoretical peak; speedup vs V0.\n\n"
              << std::left << std::setw(22) << "Version"
              << std::right << std::setw(13) << "Time(us)"
              << std::setw(14) << "BW(GB/s)" << std::setw(15) << "BW Util(%)"
              << std::setw(13) << "BW Speedup" << std::setw(16) << "MaxAbsError"
              << std::setw(10) << "Check" << '\n';

    // 4. 统一预热、校验和计时。NaN/Inf 及未写入的输出均判为失败。
    auto run_version = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(d_output, 0xff, bytes));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
        bool passed = true;
        double max_error = 0.0;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(output[i])) { passed = false; continue; }
            const double error = std::abs(static_cast<double>(output[i]) - reference[i]);
            max_error = std::max(max_error, error);
            if (error > 1e-7 + 1e-4 * std::abs(reference[i])) passed = false;
        }
        for (int row = 0; row < M; ++row) {
            double sum = 0.0;
            for (int col = 0; col < N; ++col)
                sum += output[static_cast<size_t>(row) * N + col];
            if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1e-4) passed = false;
        }
        for (int i = 0; i < WARMUP; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < REPEATS; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        const double us = milliseconds * 1000.0 / REPEATS;
        if (baseline_us == 0.0) baseline_us = us;
        // 统一有效字节数以便跨版本比较；这不是硬件计数器测得的 DRAM 流量。
        const double gbps = 2.0 * bytes / (us * 1000.0);
        std::cout << std::left << std::setw(22) << name << std::right
                  << std::fixed << std::setprecision(3) << std::setw(13) << us
                  << std::setprecision(2) << std::setw(14) << gbps
                  << std::setw(15) << gbps / peak_gbps * 100.0
                  << std::setw(13) << baseline_us / us
                  << std::scientific << std::setw(16) << max_error
                  << std::setw(10) << (passed ? "PASS" : "FAIL") << '\n';
        all_passed &= passed;
    };

    // V0 一线程一行；V1/V2/V3 一 block 一行；V4 用有限 blocks 循环覆盖所有行。
    run_version("online_softmax_v0", [&] {
        online_softmax_v0<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, M, N);
    });
    run_version("online_softmax_v1", [&] {
        online_softmax_v1<<<M, BLOCK_SIZE>>>(d_input, d_output, M, N);
    });
    run_version("online_softmax_v2", [&] {
        online_softmax_v2<<<M, BLOCK_SIZE>>>(d_input, d_output, M, N);
    });
    run_version("online_softmax_v3", [&] {
        online_softmax_v3<<<M, BLOCK_SIZE>>>(d_input, d_output, M, N);
    });
    run_version("online_softmax_v4", [&] {
        online_softmax_v4<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, M, N);
    });

    // 5. 释放资源，校验失败时返回非零状态。
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
