#include <iostream>
#include <vector>
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

// 策略1 分层归约减少冲突 分层聚合
__global__ void hierarchical_reduce(
    float* input,
    float* output,
    int N
) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;

    // 第一层：warp内规约 不用原子操作
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);// warp内线程通过寄存器直接访问
    }
    
    // 第二层：每个warp的lane0原子加到共享内存
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&block_sum, val);
    }
    __syncthreads();
    
    // 第三层：每个block的 thread 0原子加到全局结果
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// 策略二：共享内存原子代替全局内存原子
__global__ void histogram_shared(
    int* data, 
    int* hist,
    int N,
    int numBins
) {
    __shared__ int local_hist[256];

    if (threadIdx.x < numBins) {// 初始化桶
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // 在共享内存做原子加法
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&local_hist[data[tid]], 1);
    }
    __syncthreads();

    // 写回全局内存
    if (threadIdx.x < numBins) {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}

// 策略三：私有化（Privatization）
__global__ void privatized_histogram(
    int* data,
    int* hist,
    int N
) {
    // 每个线程维护私有计数器（适用于 bin 数很少的情况）
    int private_hist[4] = {0, 0, 0, 0};
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride) {
        private_hist[data[i]]++;
    }
    for (int bin = 0; bin < 4; bin++) {
        if (private_hist[bin] > 0) {
            atomicAdd(&hist[bin], private_hist[bin]);
        }
    }
}

//  策略四：Warp 聚合原子（CUDA 9+）
__device__ int atomicAggInc(
    int* cnt
) {
    // 找出 Warp 内所有要做 +1 的线程
    unsigned int active = __activemask();
    int leader = __ffs(active) - 1;  // 选出领导者（最低位的活跃线程）
    int lane = threadIdx.x % 32;

    int num_peers = __popc(active);  // 活跃线程数

    int old_val;
    if (lane == leader) {
        old_val = atomicAdd(cnt, num_peers);  // 一次原子操作代替多次
    }
    old_val = __shfl_sync(active, old_val, leader);

    // 每个线程得到自己的唯一递增值
    int rank = __popc(active & ((1 << lane) - 1));  // 在活跃线程中的排名
    return old_val + rank;
}


int main () {
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int N = 1 << 20;
    constexpr int bins = 4;
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    std::vector<float> h_input(N, 1.0f);
    float* d_input = nullptr;
    float* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    hierarchical_reduce<<<blocks, threads>>>(d_input, d_sum, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    if (h_sum != static_cast<float>(N)) {
        std::cerr << "hierarchical_reduce failed: " << h_sum << "\n";
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum));

    std::vector<int> h_data(N), h_hist(bins, 0);
    for (int i = 0; i < N; ++i) h_data[i] = i % bins;
    int *d_data = nullptr, *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(int)));
    histogram_shared<<<blocks, threads>>>(d_data, d_hist, N, bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost));
    for (int count : h_hist) {
        if (count != N / bins) {
            std::cerr << "histogram_shared failed\n";
            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(int)));
    privatized_histogram<<<blocks, threads>>>(d_data, d_hist, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost));
    for (int count : h_hist) {
        if (count != N / bins) {
            std::cerr << "privatized_histogram failed\n";
            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    std::cout << "All atomicAdd strategies passed (sum=" << h_sum
              << ", histogram bins=" << bins << ").\n";
    return 0;
}
