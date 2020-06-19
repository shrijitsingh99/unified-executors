#pragma once

#include <cuda_runtime_api.h>

template <typename Function>
__global__ void global_kernel(Function f);
