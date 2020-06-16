//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#ifdef CUDA
#include <cuda_runtime_api.h>
#endif

#include <execution/executor/base_executor.hpp>

template <typename Interface, typename Cardinality, typename Blocking,
    typename ProtoAllocator>
struct cuda_executor;

#ifdef CUDA
template <> struct execution::executor_available<cuda_executor> : std::true_type {};
#endif

template <typename Interface,typename Cardinality,typename Blocking, typename ProtoAllocator>
struct cuda_executor: executor<sse_executor, Interface, Cardinality, Blocking, ProtoAllocator> {

  using shape_type = std::array<int, 6>;

  template <typename F, typename... Args>
  void bulk_execute(F &&f, shape_type shape, Args &&... args) {
#ifdef CUDA
    void *kernel_args[] = {&args...};
    dim3 grid_size(shape[0], shape[1], shape[2]);
    dim3 block_size(shape[3], shape[4], shape[5]);
    cudaLaunchKernel(static_cast<void*>(f), grid_size, block_size, kernel_args, 0, 0);
    cudaDeviceSynchronize();
#endif
  }

  auto decay_t() -> decltype(auto) {
    if constexpr (execution::is_executor_available_t<cuda_executor>()) {
      return *this;
    }
    else
      return inline_executor<oneway_t, single_t, blocking_t::always_t, ProtoAllocator>{};
  }

  static std::string name() { return "cuda"; }
};

