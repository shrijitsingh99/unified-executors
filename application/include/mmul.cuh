//
// Created by shrijit on 5/31/20.
//

#pragma once

#include <stdexcept>

#ifdef CUDA
__global__
#endif
void mmul_gpu(double *a, double* b, double* c, int m, int n, int k);

#ifdef CUDA
__host__ void* device_upload(void *var, std::size_t size);
#else
void* device_upload(void *var, std::size_t size) {
  throw std::runtime_error("No GPU");
}
#endif