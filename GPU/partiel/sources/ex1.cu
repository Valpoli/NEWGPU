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
            // total += y[k + P] * x[i+k]
            total += y[0] * x[0];
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
    int *res = (int*) malloc(sizeof(int) * x.size());
    cudaMemcpy(res, res_GPU, x.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> res_vec(x.size());

    for (int k = 0; k < x.size(); ++k) {
        res_vec[k] = res[k];
    }
    cudaFree(x_GPU);
    cudaFree(y_GPU);
    cudaFree(res_GPU);
    return res_vec;
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
