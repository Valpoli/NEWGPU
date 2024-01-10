#include "ex2.h"

__global__
void kernel_matvecmul1(int* A, int* b, int N, int M, int* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        int result = 0;
        for (int k = 0; k < M; ++k) {
            int element_A = A[j + i * M];
            int element_B = b[j];
            result += element_A * element_B;
        }
        c[i] = result;
    }
}

std::vector<int> matvecmul1(const std::vector<int>& A, const std::vector<int>& b)
{
    const int M = b.size();    
    const int N = A.size() / M;

    const dim3 threads_per_block(BLOCK_SIZE,BLOCK_SIZE,1);
    const dim3 blocks((N + BLOCK_SIZE -1)/BLOCK_SIZE, (M + BLOCK_SIZE -1)/BLOCK_SIZE, 1);

    int *d_A;
    int *d_B;
    int *res_GPU;

    cudaMalloc(&d_A, N * M * sizeof(int));
    cudaMalloc(&d_B, M * sizeof(int));
    cudaMalloc(&res_GPU, N * sizeof(int));

    cudaMemcpy(d_A, A.data(), N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), M * sizeof(int), cudaMemcpyHostToDevice);

    kernel_matvecmul1<<<blocks,threads_per_block>>>(d_A,d_B,N,M,res_GPU);


    int *res = (int*) malloc(sizeof(int) * N);
    cudaMemcpy(res, res_GPU, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> res_vec(N);

    for (int k = 0; k < N; ++k) {
        res_vec[k] = res[k];
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(res_GPU);
    return res_vec;

}



__global__
void kernel_matvecmul2(int* A, int* b, int N, int M, int* c)
{
    // ...
}

std::vector<int> matvecmul2(const std::vector<int>& A, const std::vector<int>& b)
{
    const int M = b.size();    
    const int N = A.size() / M;

    // ...

    return {};
}
