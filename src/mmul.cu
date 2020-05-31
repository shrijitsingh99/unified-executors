#include <iostream>
#include <cuda.h>
#include <mmul_gpu.h>

__global__ void mmul_gpu()
{
  printf("mmul_gpu Called\n");
}
