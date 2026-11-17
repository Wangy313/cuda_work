#include <iostream>
#include <iomanip>
#include <functional>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cpu_baseline
void cpu_baseline(
    float* input,
    float* output,
    int N
) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = sum;
}

// gpu_baesline
__global__ void gpu_reduce_baesline(
    float* input,
    float* output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(output, input[tid]);
    }
} 

// Reduce_v0
__global__ void reduce_v0(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];// 动态共享内存
    
    int tid = threadIdx.x;// 块内线程id
    int gid = blockIdx.x * blockDim.x + threadIdx.x;// 全局线程id
    
    // 全局内存 -> 共享内存
    smem[tid] = gid < N ? input[gid] : 0.0f;
    __syncthreads();

    // 树形规约 从小到大
    for (int step = 1; step < blockDim.x; step <<= 1) {
        if (tid % (2 * step) == 0) {// 只让步长的倍数的线程工作
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 每个block内的第一个线程就是最终的部分结果
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Reduce_v1
__global__ void reduce_v1(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];// 动态共享内存
    
    int tid = threadIdx.x;// 块内线程id
    int gid = blockIdx.x * blockDim.x + threadIdx.x;// 全局线程id
    
    // 全局内存 -> 共享内存
    smem[tid] = gid < N ? input[gid] : 0.0f;
    __syncthreads();

    // 树形规约 从小到大
    for (int step = 1; step < blockDim.x; step <<= 1) {
        int idx = 2 * step * tid;// 直接把用的算出来
        if (idx < blockDim.x) {
            smem[idx] += smem[idx + step];
        }
        __syncthreads();
    }

    // 每个block内的第一个线程就是最终的部分结果
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Reduce_v2
__global__ void reduce_v2(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // input -> smem
    smem[tid] = gid < N ? input[gid] : 0.0f;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Reduce_v3
__global__ void reduce_v3(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 每个线程加载2个相距blockDim.x的元素并求和
    float val = 0.0f;
    if (gid < N) {
        val += input[gid];
    }
    if (gid + blockDim.x < N) {
        val += input[gid + blockDim.x];
    }
    smem[tid] = val;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[tid];
    }
}

// 注意：warpReduce 中的 smem 必须声明为 volatile，
// 防止编译器将中间结果缓存到寄存器，导致其他线程读取到旧值。这是保证 Warp 内线程间数据可见性的关键。
__device__ void warpReduce(
    volatile float* smem, 
    int tid
) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

// Reduce_v4 - v3基础上展开最后一个warp
__global__ void reduce_v4(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (gid < N) {
        val += input[gid];
    }
    if (gid + blockDim.x < N) {
        val += input[gid + blockDim.x];
    }
    smem[tid] = val;
    __syncthreads();

    // 规约循环执行到step > 32
    for (int step = blockDim.x / 2; step > 32; step >>= 1) {
        if (tid < step) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 最后一个warp内的规约
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Reduce_v5 完全循环展开（模板参数）
template <int BLOCK_SIZE>
__global__ void reduce_v5(
    float* input,
    float* output,
    int N
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    // 每个线程处理两个元素
    float val = 0.0f;
    if (gid < N) {  
        val += input[gid];
    }
    if (gid + BLOCK_SIZE < N) {
        val += input[gid + BLOCK_SIZE];
    }
    smem[tid] = val;
    __syncthreads();

    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            smem[tid] += smem[tid + 256];
            __syncthreads();
        }
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            smem[tid] += smem[tid + 128];
            __syncthreads();
        }
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            smem[tid] += smem[tid + 64];
            __syncthreads();
        }
    }

    // 最后 Warp 内展开
    if (tid < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Warp内部规约函数
__device__ float warpReduce(
    float val
) {
    for (int step = 16; step > 0; step >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, step);
    }
    return val;
}

// Reduce_v6 Warp Shuffle 两级规约（Warp内 + Warp间） 替代 Shared Memory  
// __shfl_down_sync：每个线程读取 lane_id+delta 处线程的寄存器值（下移操作）
// float val = __shfl_down_sync(0xffffffff, var, delta);
__global__ void reduce_v6(
    float* input,
    float* output,
    int N
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;// 线程在warp里的编号
    int wid = tid / 32;// 属于哪个warp，warp编号

    // 每个线程处理2个元素
    float val = 0.0f;
    if (gid < N) {
        val += input[gid];
    }
    if (gid + blockDim.x < N) {
        val += input[gid + blockDim.x];
    }

    // 第一级：Warp内规约
    val = warpReduce(val);

    // 第二级：Warp间规约
    // 2.1 把lane0的值放入 共享内存
    __shared__ float warp_smem[32];// 最多开1024个线程，1024/32=32个warp
    if (lane == 0) {
        warp_smem[wid] = val;
    }
    __syncthreads();

    // 2.2 warp之间规约
    int num_warp = blockDim.x / 32;// 实际warp数量
    if (wid == 0) {// 只让第一个warp里的线程去共享内存里取
        val = lane < num_warp ? warp_smem[lane] : 0.0f;
        val = warpReduce(val);
    }
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// Reduce_v7 float4 向量化加载 + Grid Stride Loop + Warp Shuffle

__global__ void reduce_v7(
    float* input,
    float* output,
    int N
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    
    // float4 加载：每线程每次处理4个float
    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = N / 4;  // float4 的元素数量

    float val = 0.0f;

    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n4; idx += gridDim.x * blockDim.x) {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w; 
    }

    // 处理n不是4的倍数的尾部元素
    int tail_start = n4 * 4;
    // 绝对索引-物理起点+线程相对位置-所有线程都+tail_start
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid; idx < N; idx += gridDim.x * blockDim.x) {
        val += input[idx];
    }

    // Warp内规约
    val = warpReduce(val);
    // for (int step = 16; step > 0; step >>= 1) {
    //     val += __shfl_down_sync(0xFFFFFFFF, val, step);
    // }

    __shared__ float warp_smem[32];
    if (lane == 0) {
        warp_smem[wid] = val;
    }
    __syncthreads();
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = lane < num_warps ? warp_smem[lane] : 0.0f;
        val = warpReduce(val);
        // for (int step = 16; step > 0; step >>= 1) {
        //     val += __shfl_down_sync(0xFFFFFFFF, val, step);
        // }
    }
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// =====================================================
// CPU 主函数
// =====================================================
__global__ void finalize_reduce(const float* partial, float* output, int N) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float value = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) value += partial[i];
    smem[tid] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) smem[tid] += smem[tid + step];
        __syncthreads();
    }
    if (tid == 0) *output = smem[0];
}

struct BenchmarkResult {
    std::string name;
    float time_us;
    float speedup;
    float bandwidth_gbps;
    float utilization;
};

int main() {
    constexpr int N = 1 << 24;
    constexpr int BLOCK = 256;
    constexpr int REPEATS = 5;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double peak_bw = static_cast<double>(prop.memoryClockRate) * 1000.0 *
                           2.0 * static_cast<double>(prop.memoryBusWidth) / 8.0;
    const int blocks_one = (N + BLOCK - 1) / BLOCK;
    const int blocks_two = (N + 2 * BLOCK - 1) / (2 * BLOCK);
    float *d_input = nullptr, *d_partial = nullptr, *d_output = nullptr;
    void *d_cub_storage = nullptr;
    size_t cub_storage_bytes = 0;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, blocks_one * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cub::DeviceReduce::Sum(d_cub_storage, cub_storage_bytes,
                                     d_input, d_output, N));
    CUDA_CHECK(cudaMalloc(&d_cub_storage, cub_storage_bytes));
    std::vector<float> input(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<BenchmarkResult> results;
    bool all_ok = true;

    auto measure = [&](const std::string& name, int partial_count, double bytes,
                       const std::function<void()>& launch) {
        float answer = 0.0f;
        float total_us = 0.0f;
        for (int r = 0; r < REPEATS + 1; ++r) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            if (r == 0) continue;
            total_us += elapsed_ms * 1000.0f;
            CUDA_CHECK(cudaMemcpy(&answer, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        }
        if (std::fabs(answer - static_cast<float>(N)) > 1.0f) {
            std::cerr << name << " result mismatch: " << answer << "\n";
            all_ok = false;
            return;
        }
        const float time_us = total_us / REPEATS;
        const double bw = bytes / (time_us * 1e-6);
        results.push_back({name, time_us, 0.0f, static_cast<float>(bw / 1e9),
                           static_cast<float>(bw / peak_bw)});
        (void)partial_count;
    };

    measure("gpu_reduce_baesline", 1, 2.0 * N * sizeof(float), [&] {
        gpu_reduce_baesline<<<blocks_one, BLOCK>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());
    });

    auto partial = [&](const std::string& name, int count, const std::function<void()>& first) {
        const double bytes = static_cast<double>(N) * sizeof(float) +
                             2.0 * count * sizeof(float) + sizeof(float);
        measure(name, count, bytes, [&] {
            first();
            CUDA_CHECK(cudaGetLastError());
            finalize_reduce<<<1, BLOCK, BLOCK * sizeof(float)>>>(d_partial, d_output, count);
            CUDA_CHECK(cudaGetLastError());
        });
    };
    partial("reduce_v0", blocks_one, [&] { reduce_v0<<<blocks_one, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v1", blocks_one, [&] { reduce_v1<<<blocks_one, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v2", blocks_one, [&] { reduce_v2<<<blocks_one, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v3", blocks_two, [&] { reduce_v3<<<blocks_two, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v4", blocks_two, [&] { reduce_v4<<<blocks_two, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v5", blocks_two, [&] { reduce_v5<BLOCK><<<blocks_two, BLOCK, BLOCK * sizeof(float)>>>(d_input, d_partial, N); });
    partial("reduce_v6", blocks_two, [&] { reduce_v6<<<blocks_two, BLOCK>>>(d_input, d_partial, N); });
    const int blocks_v7 = std::min(blocks_two, prop.multiProcessorCount * 8);
    partial("reduce_v7", blocks_v7, [&] { reduce_v7<<<blocks_v7, BLOCK>>>(d_input, d_partial, N); });
    measure("cub::DeviceReduce", 1, static_cast<double>(N) * sizeof(float) + sizeof(float), [&] {
        CUDA_CHECK(cub::DeviceReduce::Sum(d_cub_storage, cub_storage_bytes,
                                          d_input, d_output, N));
    });

    // Use reduce_v0 as the comparison baseline for all relative speedups.
    const auto baseline_it = std::find_if(results.begin(), results.end(),
                                          [](const BenchmarkResult& r) {
                                              return r.name == "reduce_v0";
                                          });
    if (baseline_it == results.end()) {
        std::cerr << "reduce_v0 baseline result is missing\n";
        return EXIT_FAILURE;
    }
    const float baseline_us = baseline_it->time_us;
    std::cout << std::fixed << std::setprecision(3)
              << "N=" << N << ", block=" << BLOCK
              << ", peak bandwidth=" << peak_bw / 1e9 << " GB/s\n\n";
    std::cout << std::left << std::setw(22) << "Function"
              << std::right << std::setw(14) << "Time(us)"
              << std::setw(14) << "V0Speedup"
              << std::setw(16) << "BW(GB/s)"
              << std::setw(16) << "BW util(%)" << "\n";
    auto print_result = [&](BenchmarkResult& result) {
        result.speedup = baseline_us / result.time_us;
        std::cout << std::left << std::setw(22) << result.name
                  << std::right << std::setw(14) << result.time_us
                  << std::setw(14) << result.speedup
                  << std::setw(16) << result.bandwidth_gbps
                  << std::setw(15) << result.utilization * 100.0f << "%\n";
    };
    print_result(*baseline_it);
    for (auto& result : results) {
        if (&result != &*baseline_it) print_result(result);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_cub_storage));
    return all_ok ? 0 : EXIT_FAILURE;
}
