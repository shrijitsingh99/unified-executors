//
// Created by shrijit on 5/31/20.
//
#include <cuda.h>

#pragma once

#ifdef __CUDACC__
__global__ void mmul_gpu();
#else
void mmul_gpu();
#endif