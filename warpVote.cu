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
// 所有线程的 predicate 都为 true 时返回 true
int __all_sync(unsigned mask, int predicate);

// 任意线程的 predicate 为 true 时返回 true
int __any_sync(unsigned mask, int predicate);

// 返回一个 32-bit 位图，每个 bit 对应一个 lane 的 predicate 值
unsigned __ballot_sync(unsigned mask, int predicate);

__activemask()：点名册   
__activemask() 会返回一个 32 位的整数，哪一位是 1，就代表那个编号的线程当前就在现场。

__syncwarp()：小分队集合令
Warp 内部同步屏障。它会强制指定的线程必须全部走到这一行，才能继续往下走。

// 返回具有相同 val 值的线程掩码
unsigned __match_any_sync(unsigned mask, T val);

// 如果所有参与线程的 val 相同，返回全体掩码；否则返回 0
unsigned __match_all_sync(unsigned mask, T val, int* pred);


*/

// 提前退出优化
__global__ void search_kernel(float* data, float target, float* found) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pred = (data[tid] == target);
    if (__any_sync(0xFFFFFFFF, pred)) {
        if (pred) {
            atomicExch(found, tid);
        }
    }
}

int main() {

    return 0;
}