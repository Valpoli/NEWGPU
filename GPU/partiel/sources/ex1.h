#pragma once

#include "common.h"

#define BLOCK_SIZE 16 // number of threads per block


__global__
void kernel_conv1(int* x, int* y, int N, int M, int* z);

std::vector<int> conv1(const std::vector<int>& x, const std::vector<int>& y);


__global__
void kernel_conv2(int* x, int* y, int N, int M, int* z);

std::vector<int> conv2(const std::vector<int>& x, const std::vector<int>& y);

