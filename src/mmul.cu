#include "mmul.cuh"

__global__ void mmul_gpu(double *a, double *b, double *c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0;
  if (col < k && row < m) {
    for (int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

__host__ void* device_upload(void *var, std::size_t size) {
  void *gpu_var;
  cudaMallocManaged(&gpu_var, size);
  memcpy(gpu_var, var, size);
  return gpu_var;
}
