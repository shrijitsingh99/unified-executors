#include "cuda_executor.cuh"

template <typename Function>
__global__ void global_kernel(Function f) {
  f();
}
