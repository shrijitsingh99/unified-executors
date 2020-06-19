//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#ifdef CUDA
#include <cuda_runtime_api.h>

#include <execution/cuda_executor.cuh>
#endif

#include <execution/executor/base_executor.hpp>
#include <execution/executor/inline_executor.hpp>

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator>
struct cuda_executor;

#ifdef CUDA
namespace execution {
template <>
struct is_executor_available<cuda_executor> : std::true_type {};
}  // namespace execution
#endif

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator = std::allocator<void>>
struct cuda_executor : executor<cuda_executor, Interface, Cardinality, Blocking,
                                ProtoAllocator> {
  using shape_type = typename std::array<int, 6>;

  template <typename F>
  void execute(F &&f) {
#ifdef CUDA
    cudaLaunchKernel(static_cast<void *>(&f), 1, 1, nullptr, 0, 0);
    cudaDeviceSynchronize();
#endif
  }

  // Temporary fix for unit test compilation
  template <typename F>
  void bulk_execute(F &&f, std::size_t n) {
    bulk_execute(f, std::array<int, 6>{1, 1, 1, static_cast<int>(n), 1, 1});
  }

  template <typename F>
  void bulk_execute(F &&f, shape_type shape) {
#ifdef CUDA
    void *kernel_args[] = {};
    dim3 grid_size(shape[0], shape[1], shape[2]);
    dim3 block_size(shape[3], shape[4], shape[5]);
    cudaLaunchKernel(static_cast<void *>(&f), grid_size, block_size,
                     kernel_args, 0, 0);
    cudaDeviceSynchronize();
#endif
  }

  //  auto decay_t() -> decltype(auto) {
  //    if constexpr (execution::is_executor_available_v<cuda_executor>) {
  //      return *this;
  //    } else
  //      return inline_executor<oneway_t, single_t, blocking_t::always_t,
  //                             ProtoAllocator>{};
  //  }

  std::string name() { return "cuda"; }
};
