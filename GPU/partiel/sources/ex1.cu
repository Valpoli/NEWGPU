#include "ex1.h"

__global__ 
void kernel_conv1(int* x, int* y, int N, int M, int* z)
{
    const int P = (M-1) / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 0;
     for (int k = -P; k <= P; ++k) {
        if (i + k >= 0 && i + k < N)
        {
            total += y[i+k] * x[i+k];
        }
     }
    z[i] = total;
}

std::vector<int> conv1(const std::vector<int>& x, const std::vector<int>& y)
{
    // ...
    return {};
}



__global__ 
void kernel_conv2(int* x, int* y, int N, int M, int* z)
{
    const int P = (M-1) / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int s_x[BLOCK_SIZE];

    // ...
}

std::vector<int> conv2(const std::vector<int>& x, const std::vector<int>& y)
{
    // ...
    return {};
}
