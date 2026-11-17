#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/*
    BM = 128;
    BN = 128;
    BK = 8;
    TM = 8;
    TN = 8;
*/

// A: M x K，B: K x N，C: M x N，均按 row-major 连续存储。
__global__ void gemm_baseline(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程负责的 C 行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程负责的 C 列
    if (row < M && col < N) {
        float sum = 0.0f; // C[row][col] 的累加器
        for (int k = 0; k < K; k++) { // 沿公共维度 K 扫描的索引
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Gemm_v0 - Thread Block 级 Tiling 优化 - 沿 K 维度切分
// BM,BN=128 - 算术强度 32，足以饱和计算单元
// BK=8 - Shared Memory 占用可控（128×8×4=4KB/矩阵）
// BLOCK_SIZE = 256 - 匹配 BM/BN 划分，支持 2 blocks/SM
template<int BM, int BN, int BK, int BLOCK_SIZE>// 分块大小、block大小
__global__ void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
        线程重排的索引计算：
        int tid = threadIdx.x;
        // 加载 tileA 时的线程坐标（8列×32行）
        int a_thread_x = tid % 8;       // 列方向
        int a_thread_y = tid / 8;       // 行方向
        // 加载 tileB 时的线程坐标（32列×8行）
        int b_thread_x = tid % 32;      // 列方向
        int b_thread_y = tid / 32;      // 行方向

        // 计算 tileC 时的线程排布（16×16）
        constexpr int C_BLOCK_X = 16;
        constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
        int c_thread_x = tid % C_BLOCK_X;
        int c_thread_y = tid / C_BLOCK_X;

        // 每个线程负责 Tm×Tn 个输出元素
        constexpr int Tm = BM / C_BLOCK_Y;  // = 8
        constexpr int Tn = BN / C_BLOCK_X;  // = 8
        float Ct[Tm][Tn] = {0.0f};
    */
   __shared__ float Asmem[BM][BK]; // 当前 K tile 的 A 子块
   __shared__ float Bsmem[BK][BN]; // 当前 K tile 的 B 子块

   // 计算出当前这个线程块（Block）在巨大的输出矩阵 C 中，应该负责计算的“起始行（row 0）”和“起始列（column 0）”。
   int r0 = blockIdx.y * BM; // 当前 block 在 C 中覆盖区域的起始行
   int c0 = blockIdx.x * BN; // 当前 block 在 C 中覆盖区域的起始列
   int tid = threadIdx.x;    // block 内一维线程编号

   // 加载tileA的线程重排
   constexpr int A_BLOCK_X = BK; // 加载 A tile 时的线程布局宽度
   constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // 加载 A tile 时的布局高度
   int a_thread_x = tid % A_BLOCK_X; // 线程在 A tile 中的列坐标
   int a_thread_y = tid / A_BLOCK_X; // 线程在 A tile 中的行坐标

   // 加载tileB的线程重排
   constexpr int B_BLOCK_X = 32; // 一个 warp 负责 B tile 的一段列
   constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // 加载 B tile 时的布局高度
   int b_thread_x = tid % B_BLOCK_X; // 线程在 B tile 中的列坐标
   int b_thread_y = tid / B_BLOCK_X; // 线程在 B tile 中的行坐标

   // 计算tileC的线程重排序（16*16）
   constexpr int C_BLOCK_X = BN; // 计算 tile C 时的线程布局宽度
   constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X; // 计算 tile C 时的布局高度
   int c_thread_x = tid % C_BLOCK_X; // 线程在 C tile 中的列坐标
   int c_thread_y = tid / C_BLOCK_X; // 线程在 C tile 中的行坐标

   // 每个线程负责Tm*Tn个输出元素
   constexpr int Tm = BM / C_BLOCK_Y; // 每个线程负责的 C 行数
   constexpr int Tn = BN / C_BLOCK_X; // 每个线程负责的 C 列数
   float Ct[Tm][Tn] = {0.0f}; // 线程私有的输出累加器

   // K-Loop
    for (int k_tile = 0; k_tile < K; k_tile += BK) {// 沿 K 维度遍历 tile
        // 加载tileA（使用跨步循环覆盖BM行）
        #pragma unroll// 循环展开加速
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = r0 + i; // A tile 中的全局行
            int c = k_tile + a_thread_x; // A tile 中的全局列
            Asmem[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;// 放入共享内存
        }

        // 加载 tile B：线程在 K 维和 N 维共同覆盖整个 BK x BN tile
        #pragma unroll
        for (int i = b_thread_y; i < BK; i += B_BLOCK_Y) {
            for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
                int r = k_tile + i; // B tile 中的全局行
                int c = c0 + j; // B tile 中的全局列
                Bsmem[i][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
            }
        }
        __syncthreads();

        // 外积方式计算 As * Bs
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) { // 当前 tile 内的 K 坐标
            for (int i = 0; i < Tm; i++) {
                int r = c_thread_y + i * C_BLOCK_Y; // A/C tile 行坐标
                for (int j = 0; j < Tn; j++) {
                    int c = c_thread_x + j * C_BLOCK_X; // B/C tile 列坐标
                    Ct[i][j] += Asmem[r][kk] * Bsmem[kk][c];
                }
            }
        }
        __syncthreads();
    }
    // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y; // C 的全局行
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X; // C 的全局列
            if (c < N && r < M) {
                C[r * N + c] = Ct[i][j]; // row-major 的 C 行步长是 N
            }
        }
    }
}

// Gemm_v1 -  Thread 级 Tiling 优化：寄存器级数据复用
template <int BM, int BN, int BK, int BLOCK_SIZE> 
__global__ void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float Asmem[BM][BK]; // 当前线程块缓存的 A 子矩阵
    __shared__ float Bsmem[BK][BN]; // 当前线程块缓存的 B 子矩阵

    // 1. 补齐缺失的基础坐标和线程排布
    int r0 = blockIdx.y * BM; // 当前线程块在 C 中的起始行
    int c0 = blockIdx.x * BN; // 当前线程块在 C 中的起始列
    int tid = threadIdx.x; // 当前线程在线程块内的一维编号

    // 用于加载 A 的线程索引 (假设 A 的宽度为 BK)
    constexpr int A_BLOCK_X = BK;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    int a_thread_x = tid % A_BLOCK_X; // 加载 A 时在线程布局中的列坐标
    int a_thread_y = tid / A_BLOCK_X; // 加载 A 时在线程布局中的行坐标

    // 用于加载 B 的线程索引 (假设 B 的宽度为 BN)
    constexpr int B_BLOCK_X = BN;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
    int b_thread_x = tid % B_BLOCK_X; // 加载 B 时在线程布局中的列坐标
    int b_thread_y = tid / B_BLOCK_X; // 加载 B 时在线程布局中的行坐标

    // 2. 计算 C 的线程分布
    constexpr int C_BLOCK_X = BN;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;
    constexpr int TM = BM / C_BLOCK_Y;
    constexpr int TN = BN / C_BLOCK_X;

    // 每个线程在 C 中负责 TM×TN 的子块
    int thread_row = (tid / C_BLOCK_X) * TM; // 当前线程负责的 C 子块起始行
    int thread_col = (tid % C_BLOCK_X) * TN; // 当前线程负责的 C 子块起始列

    // 寄存器存储 
    float a_frag[TM]; // 当前线程从 A 共享内存加载的寄存器片段
    float b_frag[TN]; // 当前线程从 B 共享内存加载的寄存器片段
    float c_frag[TM][TN] = {0.0f}; // 当前线程在寄存器中累加的 C 子块

    int by = blockIdx.y, bx = blockIdx.x; // 当前线程块在二维网格中的行列编号

    for (int bk = 0; bk < K; bk += BK) { // 当前处理的 K 维 tile 起点
        // 加载tileA（使用跨步循环覆盖BM行）
        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {
            int r = r0 + i;         // A tile 中的全局行
            int c = bk + a_thread_x; // A tile 中的全局列
            Asmem[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

        // 加载tileB（使用跨步循环覆盖BN列）
        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = bk + b_thread_y; // B tile 中的全局行
            int c = c0 + j;          // B tile 中的全局列
            Bsmem[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

        // 外积累加 (内层循环变量改为 k_inner，避免与外层的 bk 混淆)
        for (int k_inner = 0; k_inner < BK; k_inner++) { // 当前 tile 内的 K 坐标
            // 从 Shared Memory 加载到寄存器 (修正了 Asmem 和 Bsmem 的拼写)
            for (int i = 0; i < TM; i++) {
                a_frag[i] = Asmem[thread_row + i][k_inner]; // A 的寄存器片段元素
            }
            for (int j = 0; j < TN; j++) {
                b_frag[j] = Bsmem[k_inner][thread_col + j]; // B 的寄存器片段元素
            }
            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回 Global Memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            C[(by * BM + thread_row + i) * N + bx * BN + thread_col + j] = c_frag[i][j];
        }
    }
}

// =====================================================
// 1. Global Memory -> Shared Memory
// =====================================================
// 协作加载一个 K Tile。
// A 在 Shared Memory 中采用 [BK][BM] 布局。
// B 在 Shared Memory 中采用 [BK][BN] 布局。
// 在共享内存中转置存储为了解决bank conflict，同时支持向量化读取
// __forceinline__ 强制编译器把这个函数的代码嵌入调用的地方，降低函数调用开销
template<int BM, int BN, int BK, int BLOCK_SIZE>
__device__ __forceinline__ void load_global_to_shared(
    const float* A,
    const float* B,
    float Asmem[BK][BM],
    float Bsmem[BK][BN], 
    int M,
    int N,
    int K,
    int bk
) {
    int tid = threadIdx.x;
    // 加载A
    // 线程布局
    constexpr int A_BLOCK_X = BK;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    // 线程映射
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;
    // 协作加载BM*BN个元素
    #pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {// 合并访问
        // 数据全局坐标
        int r = blockIdx.y * BM + i;
        int c = bk + a_thread_x;
        // 加载
        Asmem[a_thread_x][i] = (r < M && c < K) ? A[r * K + c] : 0.0f;
    }
    // 加载B
    // 线程布局
    constexpr int B_BLOCK_X = 32;// 正好一个warp   
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
    // 线程映射
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;
    #pragma unroll
    for (int i = b_thread_y; i < BK; i += B_BLOCK_Y) {
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            // 全局内存坐标映射
            int r = bk + i;
            int c = blockIdx.x * BN + j;
            // 加载
            Bsmem[i][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
}

// =====================================================
// Shared Memory -> Register
// 加载当前 K 位置的 A、B 数据片段
// =====================================================
template<int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void load_shared_to_reg(
    const float Asmem[BK][BM],
    const float Bsmem[BK][BN],
    float a_frag[TM],
    float b_frag[TN],
    int k
) {
    int tid = threadIdx.x;

    // 一行的线程数量
    constexpr int THREADS_PER_ROW  = BN / TN;

    int thread_row = (tid / THREADS_PER_ROW) * TM;
    int thread_col = (tid % THREADS_PER_ROW) * TN;

    // 加载A
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        a_frag[i] = Asmem[k][thread_row + i];
    }
    // 加载B
    #pragma unroll
    for (int j = 0; j < TN; j++) {
        b_frag[j] = Bsmem[k][thread_col + j];
    }
}

template<int BM, int BN, int BK, int BLOCK_SIZE> 
__device__ __forceinline__ void issue_ldg(
    const float* A,
    const float* B,
    float ldg_a[4],
    float ldg_b[4],
    int M,
    int N,
    int K,
    int tile_k
) {
    int tid = threadIdx.x;
    // 1. 加载 Tile A：Global Memory -> Register
    constexpr int A_BLOCK_X = BK;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;
    // 每个线程加载4个元素
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int i = a_thread_y + q * A_BLOCK_Y;
        if (i < BM) {
            int r = blockIdx.y * BM + i;
            int c = tile_k + a_thread_x;
            ldg_a[q] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        } else {
            ldg_a[q] = 0.0f;
        }
    }
    // 2. 加载 Tile B：Global Memory -> Register
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;
    // 每个线程加载 4 个 B 元素
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int p = b_thread_y;
        int j = b_thread_x + q * B_BLOCK_X;
        if (p < BK && j < BN) {
            int r = tile_k + p;
            int c = blockIdx.x * BN + j;
            ldg_b[q] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        } else {
            ldg_b[q] = 0.0f;
        }
    }
}

// =====================================================
// Register -> Register
// 外积计算：C_frag += A_frag × B_frag
// =====================================================

template<int TM, int TN> 
__device__ __forceinline__ void outer_product(
    const float (&a_frag)[TM],
    const float (&b_frag)[TN],
    float (&c_frag)[TM][TN]
) {
    // 外积计算
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            c_frag[i][j] += a_frag[i] * b_frag[j];
        }
    }
}

// =====================================================
// Register -> Shared Memory
// 将预取的 A、B 数据写入下一个 Shared Memory Buffer
// =====================================================

template<int BM, int BN, int BK, int BLOCK_SIZE>
__device__ __forceinline__ void store_reg_to_shared(
    const float ldg_a[4],
    const float ldg_b[4],
    float Asmem[BK][BM],
    float Bsmem[BK][BN]
) {
    int tid = threadIdx.x;

    // =================================================
    // 1. Register -> Shared Memory：写入 Tile A
    // =================================================

    // 必须与 issue_ldg() 使用相同的线程映射
    constexpr int A_BLOCK_X = BK;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;

    int a_thread_x = tid % A_BLOCK_X;
    int a_thread_y = tid / A_BLOCK_X;

    // 每个线程将 4 个 A 元素写入共享内存
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int i = a_thread_y + q * A_BLOCK_Y;
        if (i < BM) {
            Asmem[a_thread_x][i] = ldg_a[q];
        }
    }
    // =================================================
    // 2. Register -> Shared Memory：写入 Tile B
    // =================================================
    // 必须与 issue_ldg() 使用相同的线程映射
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;
    // 每个线程将 4 个 B 元素写入共享内存
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int p = b_thread_y;
        int j = b_thread_x + q * B_BLOCK_X;
        if (p < BK && j < BN) {
            Bsmem[p][j] = ldg_b[q];
        }
    }
}

// =====================================================
// Register -> Global Memory
// 将每个线程的 C 子块写回全局内存
// =====================================================

template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void store_c(
    const float (&c_frag)[TM][TN],
    float* C,
    int M,
    int N
) {
    int tid = threadIdx.x;
    // =================================================
    // 1. 计算当前线程负责的 C 子块位置
    // =================================================
    constexpr int THREADS_PER_ROW = BN / TN;
    int thread_row = (tid / THREADS_PER_ROW) * TM;
    int thread_col = (tid % THREADS_PER_ROW) * TN;
    // =================================================
    // 2. 将寄存器中的结果写回 Global Memory
    // =================================================
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            // C 矩阵中的全局行、列坐标
            int r = blockIdx.y * BM + thread_row + i;
            int c = blockIdx.x * BN + thread_col + j;
            // 边界检查
            if (r < M && c < N) {
                C[r * N + c] = c_frag[i][j];
            }
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_buffer(
    const float* A,
    const float* B, 
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int BLOCK_SIZE = (BM / TM) * (BN / TN);
    // A、B分别有两个共享内存缓冲区 - 双缓冲
    __shared__ float Asmem[2][BK][BM];
    __shared__ float Bsmem[2][BK][BN];

    // 寄存器缓冲区 - 双缓冲
    float a_frag[2][TM];
    float b_frag[2][TN];
    float c_frag[TM][TN] = {0.0f};
    float ldg_a[4];
    float ldg_b[4];// 暂存 Global Memory 加载结果

    int buf_idx = 0;

    // Prologue: 加载第一个 tile
    load_global_to_shared<BM, BN, BK, BLOCK_SIZE>(A, B, Asmem[0], Bsmem[0], M, N, K, 0);
    __syncthreads();

    // 预取第一组寄存器数据
    load_shared_to_reg<BM, BN, BK, TM, TN>(Asmem[0], Bsmem[0], a_frag[0], b_frag[0], 0);

    // 这里的 tile_k 表示准备预取的下一块数据的 K 起始位置，而当前正在计算的 Tile 已经在 Shared Memory 里。
    for (int tile_k = BK; tile_k < K + BK; tile_k += BK) {
        int next_buf = 1 - buf_idx;// 取反
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 提前把下一个 K 位置的数据加载到另一个寄存器缓冲区。
            if (k + 1 < BK) {
                load_shared_to_reg<BM, BN, BK, TM, TN>(Asmem[buf_idx], Bsmem[buf_idx], a_frag[1 - k % 2], b_frag[1 - k % 2], k + 1);
            }
            // 在第一个 k 步时，发射下一 tile 的 Global Memory 加载
            if (k == 0 && tile_k < K) {
                issue_ldg<BM, BN, BK, BLOCK_SIZE>(A, B, ldg_a, ldg_b, M, N, K, tile_k);
            }
            // 执行 TM×TN 的外积计算
            outer_product<TM, TN>(a_frag[k % 2], b_frag[k % 2], c_frag);
        }

        // 将 LDG 结果写入下一 buffer
        if (tile_k < K) {
            store_reg_to_shared<BM, BN, BK, BLOCK_SIZE>(ldg_a, ldg_b, Asmem[next_buf], Bsmem[next_buf]);
            __syncthreads();

            buf_idx = next_buf;
            load_shared_to_reg<BM, BN, BK, TM, TN>(Asmem[buf_idx], Bsmem[buf_idx], a_frag[0], b_frag[0], 0);
        }
    }

    // 写回结果
    store_c<BM, BN, TM, TN>(c_frag, C, M, N);
}


int main() {
    // ================================================================
    // 1. 设置矩阵尺寸与线程布局：A[M][K] * B[K][N] = C[M][N]
    // ================================================================
    // M、N 取 tile 的整数倍，因为当前 v1 写回时没有边界判断。
    // N 与 M 不相等，便于校验时发现输出行步长用错的问题。
    constexpr int M = 128;          // A 和 C 的行数
    constexpr int N = 192;          // B 和 C 的列数
    constexpr int K = 65;           // 公共维度；非 BK 整数倍，用于检查末尾补零
    constexpr int BM = 32;          // 一个 block 计算的 C 子矩阵行数
    constexpr int BN = 32;          // 一个 block 计算的 C 子矩阵列数
    constexpr int BK = 8;           // 每次加载的 K 维长度
    constexpr int BLOCK_SIZE = 256; // v0、v1 使用的一维 block 线程数
    constexpr int BUFFER_TM = 2;    // gemm_buffer 中每个线程负责的 C 行数
    constexpr int BUFFER_TN = 2;    // gemm_buffer 中每个线程负责的 C 列数
    constexpr int REPEATS = 10;     // 预热后用于求平均耗时的执行次数

    // 当前 kernel 的 B 加载循环只覆盖 BLOCK_SIZE / BN 行。
    // 因此选择 BLOCK_SIZE == BK * BN，保证 B tile 的所有行都被加载。
    static_assert(BLOCK_SIZE == BK * BN, "B tile must be fully loaded");
    static_assert(BLOCK_SIZE == (BM / BUFFER_TM) * (BN / BUFFER_TN),
                  "gemm_buffer block size does not match TM/TN");
    static_assert(BLOCK_SIZE % BK == 0, "A thread layout must divide evenly");
    static_assert(M % BM == 0 && N % BN == 0, "v1 requires full output tiles");
    static_assert(BM % (BLOCK_SIZE / BN) == 0, "C rows must divide evenly");

    const size_t bytes_a = size_t(M) * K * sizeof(float); // A 占用字节数
    const size_t bytes_b = size_t(K) * N * sizeof(float); // B 占用字节数
    const size_t bytes_c = size_t(M) * N * sizeof(float); // C 占用字节数
    const dim3 baseline_block(16, 16); // baseline 每个线程计算一个 C 元素
    const dim3 baseline_grid((N + 15) / 16, (M + 15) / 16);
    const dim3 tiled_grid(N / BN, M / BM); // v0、v1 的输出 tile 网格

    // ================================================================
    // 2. 准备 CPU 数据，并计算独立参考结果
    // ================================================================
    std::vector<float> h_a(size_t(M) * K); // CPU 上的输入矩阵 A
    std::vector<float> h_b(size_t(K) * N); // CPU 上的输入矩阵 B
    std::vector<float> h_c(size_t(M) * N); // 从 GPU 拷回的输出矩阵
    std::vector<double> reference(size_t(M) * N); // double 精度参考结果

    // 使用有正有负的确定性数据，方便每次运行重现同一组校验。
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = (int(i % 17) - 8) / 16.0f;
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        h_b[i] = (int(i % 13) - 6) / 16.0f;
    }
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0; // CPU 参考结果的累加器
            for (int k = 0; k < K; ++k) {
                sum += double(h_a[row * K + k]) * h_b[k * N + col];
            }
            reference[row * N + col] = sum;
        }
    }

    // ================================================================
    // 3. 定义 CUDA 错误检查与资源释放方式
    // ================================================================
    float* d_a = nullptr; // GPU 全局内存中的 A
    float* d_b = nullptr; // GPU 全局内存中的 B
    float* d_c = nullptr; // GPU 全局内存中的 C，三个版本依次复用
    cudaEvent_t start = nullptr; // GPU 计时起点
    cudaEvent_t stop = nullptr;  // GPU 计时终点

    auto check_cuda = [](cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " +
                                     cudaGetErrorString(status));
        }
    };
    // 成功或异常退出都会执行；只释放已创建的资源。
    auto cleanup = [&]() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (d_c) cudaFree(d_c);
        if (d_b) cudaFree(d_b);
        if (d_a) cudaFree(d_a);
    };

    try {
        check_cuda(cudaSetDevice(0), "选择 GPU");
        check_cuda(cudaMalloc(&d_a, bytes_a), "分配 A");
        check_cuda(cudaMalloc(&d_b, bytes_b), "分配 B");
        check_cuda(cudaMalloc(&d_c, bytes_c), "分配 C");
        check_cuda(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice), "上传 A");
        check_cuda(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice), "上传 B");
        check_cuda(cudaEventCreate(&start), "创建计时起点");
        check_cuda(cudaEventCreate(&stop), "创建计时终点");

        std::cout << "A(" << M << " x " << K << ") * B(" << K << " x " << N
                  << ") = C(" << M << " x " << N << ")\n";
        std::cout << std::left << std::setw(18) << "Version"
                  << std::setw(16) << "Time(us)"
                  << std::setw(18) << "Max abs error" << "Check\n";

        // ============================================================
        // 4. 统一执行、计时、下载和校验
        // ============================================================
        auto run_version = [&](const char* name, auto launch) {
            // 先预热，再填 NaN。若 kernel 漏写某个输出，校验能发现它。
            launch();
            check_cuda(cudaGetLastError(), "预热 kernel 启动");
            check_cuda(cudaDeviceSynchronize(), "预热 kernel 执行");
            std::fill(h_c.begin(), h_c.end(), std::numeric_limits<float>::quiet_NaN());
            check_cuda(cudaMemcpy(d_c, h_c.data(), bytes_c, cudaMemcpyHostToDevice), "初始化输出");

            // Event 只统计 GPU 执行区间，不包含分配、上传和下载。
            check_cuda(cudaEventRecord(start), "记录起点");
            for (int repeat = 0; repeat < REPEATS; ++repeat) {
                launch();
                check_cuda(cudaGetLastError(), "kernel 启动");
            }
            check_cuda(cudaEventRecord(stop), "记录终点");
            check_cuda(cudaEventSynchronize(stop), "等待 kernel 完成");
            float elapsed_ms = 0.0f; // REPEATS 次执行的总耗时，单位毫秒
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "读取 GPU 耗时");
            check_cuda(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "下载 C");

            size_t mismatches = 0; // 超出允许误差的元素数量
            double max_error = 0.0; // 全矩阵最大绝对误差
            for (size_t i = 0; i < h_c.size(); ++i) {
                const double error = std::abs(double(h_c[i]) - reference[i]);
                const double tolerance = 1e-4 + 1e-4 * std::abs(reference[i]);
                if (!std::isfinite(h_c[i]) || error > tolerance) ++mismatches;
                max_error = std::max(max_error, std::isfinite(error) ? error :
                                     std::numeric_limits<double>::infinity());
            }
            std::cout << std::left << std::setw(18) << name
                      << std::setw(16) << elapsed_ms * 1000.0f / REPEATS
                      << std::setw(18) << max_error
                      << (mismatches == 0 ? "PASS" : "FAIL") << "\n";
            if (mismatches != 0) {
                std::cerr << name << ": " << mismatches << " 个元素校验失败\n";
            }
            return mismatches == 0;
        };

        // ============================================================
        // 5. 按顺序测试三个版本（分别校验，失败不会被误报为通过）
        // ============================================================
        bool all_passed = true; // 记录所有版本是否均通过 CPU 校验
        all_passed &= run_version("gemm_baseline", [&]() {
            gemm_baseline<<<baseline_grid, baseline_block>>>(d_a, d_b, d_c, M, N, K);
        });
        all_passed &= run_version("gemm_v0", [&]() {
            gemm_v0<BM, BN, BK, BLOCK_SIZE><<<tiled_grid, BLOCK_SIZE>>>(d_a, d_b, d_c, M, N, K);
        });
        all_passed &= run_version("gemm_v1", [&]() {
            gemm_v1<BM, BN, BK, BLOCK_SIZE><<<tiled_grid, BLOCK_SIZE>>>(d_a, d_b, d_c, M, N, K);
        });
        all_passed &= run_version("gemm_buffer", [&]() {
            gemm_buffer<BM, BN, BK, BUFFER_TM, BUFFER_TN>
                <<<tiled_grid, BLOCK_SIZE>>>(d_a, d_b, d_c, M, N, K);
        });

        // ============================================================
        // 6. 释放资源，返回能够反映校验结果的退出码
        // ============================================================
        cleanup();
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "CUDA 执行失败：" << error.what() << "\n";
        cleanup();
        return 1;
    }
}
