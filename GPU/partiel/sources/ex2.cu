#include "ex2.h"

__global__
void kernel_matvecmul1(int* A, int* b, int N, int M, int* c)
{
    // ...
}

std::vector<int> matvecmul1(const std::vector<int>& A, const std::vector<int>& b)
{
    const int M = b.size();    
    const int N = A.size() / M;

    // ...

    return {};
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
