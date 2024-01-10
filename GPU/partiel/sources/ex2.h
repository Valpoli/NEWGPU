#pragma once

#include "common.h"

#define BLOCK_SIZE 8 // number of threads per block


__global__
void kernel_matvecmul1(int* A, int* b, int N, int M, int* c);

std::vector<int> matvecmul1(const std::vector<int>& A, const std::vector<int>& b);


__global__
void kernel_matvecmul2(int* A, int* b, int N, int M, int* c);

std::vector<int> matvecmul2(const std::vector<int>& A, const std::vector<int>& b);



