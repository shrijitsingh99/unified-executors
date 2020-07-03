//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#ifdef CUDA
#include <cuda_runtime_api.h>

#include <nvfunctional>
#endif

#include <execution/executor/base_executor.hpp>
#include <execution/executor/inline_executor.hpp>

#ifdef CUDA
template <typename F>
__global__ void global_kernel(F f) {
  f();
}
#endif

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
  void execute(F &f) const {
#ifdef CUDA
    void *global_kernel_args[] = {static_cast<void *>(&f)};
    cudaLaunchKernel(reinterpret_cast<void *>(global_kernel<F>), 1, 1,
                     global_kernel_args, 0, nullptr);
    cudaDeviceSynchronize();
#endif
  }

  // Temporary fix for unit test compilation
  template <typename F>
  void bulk_execute(F &f, std::size_t n) const {
    bulk_execute(f, std::array<int, 6>{1, 1, 1, static_cast<int>(n), 1, 1});
  }

  // Passing rvalue reference of function doesn't currently work with CUDA for
  // some reason
  template <typename F>
  void bulk_execute(F &f, shape_type shape) const {
#ifdef CUDA
    void *global_kernel_args[] = {static_cast<void *>(&f)};
    dim3 grid_size(shape[0], shape[1], shape[2]);
    dim3 block_size(shape[3], shape[4], shape[5]);
    cudaLaunchKernel(reinterpret_cast<void *>(global_kernel<F>), grid_size,
                     block_size, global_kernel_args, 0, nullptr);
    cudaDeviceSynchronize();
#endif
  }

  cuda_executor<oneway_t, Cardinality, Blocking, ProtoAllocator> require(
      const oneway_t &p) const {
    return {};
  }

  cuda_executor<Interface, Cardinality, blocking_t::always_t, ProtoAllocator>
  require(const blocking_t::always_t &t) const {
    return {};
  }

  static constexpr auto name() { return "cuda"; }
};
