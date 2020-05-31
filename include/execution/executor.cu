#include <iostream>
#include <cuda.h>

#include <execution/executor_gpu.h>

template<typename F>
void launch_kernel(F&& f) {
//  f<<<1, 1>>>();
  printf("launch_kernel Called");
}