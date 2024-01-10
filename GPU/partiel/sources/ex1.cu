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
    int *x_GPU;
    int *y_GPU;
    cudaMemcpy(x_GPU, x.data(), x.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y_GPU, y.data(), y.size() * sizeof(int), cudaMemcpyHostToDevice);
    int *res_GPU;
    cudaMalloc(&res_GPU, x.size() * sizeof(int));

    const dim3 threads_per_block(BLOCK_SIZE,1,1);

    const dim3 blocks((x.size()/BLOCK_SIZE) + 1, 1, 1);

    kernel_conv1<<<blocks,threads_per_block>>>(x_GPU,y_GPU,x.size(),y.size(),res_GPU);
    std::vector<int> res(x.size());

    cudaMemcpy(res, res_GPU, x.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(x_GPU);
    cudaFree(y_GPU);
    cudaFree(res_GPU);
    return res;
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
