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



int main() {

    return 0;
}