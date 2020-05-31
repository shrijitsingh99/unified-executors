#pragma once

#include <execution/property.hpp>
#include <execution/type_traits.hpp>
#include <functional>
#include <iostream>
#include <string>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

template <typename Interface, typename Blocking, typename ProtoAllocator>
struct inline_executor;
template <> struct execution::executor_available<inline_executor> : std::true_type {};

template <typename Interface, typename Blocking, typename ProtoAllocator>
struct sse_executor;
#ifdef _SSE
template <> struct execution::executor_available<sse_executor> : std::true_type {};
#endif

template <typename Interface, typename Blocking, typename ProtoAllocator>
struct omp_executor;
#ifdef _OPENMP
template <> struct execution::executor_available<omp_executor> : std::true_type {};
#endif

template <typename Interface, typename Blocking, typename ProtoAllocator>
struct cuda_executor;

#ifdef CUDA
template <> struct execution::executor_available<cuda_executor> : std::true_type {};

#endif


template <template <typename, typename, typename> typename Derived, typename Interface, typename Blocking, typename ProtoAllocator>
class executor {
 public:
  constexpr bool query(const blocking_t::always_t &t) {
    return std::is_same<Blocking, blocking_t::always_t>();
  }

  constexpr bool query(const blocking_t::never_t &t) {
    return std::is_same<Blocking, blocking_t::never_t>();
  }

  constexpr bool query(const oneway_t &t) {
    return std::is_same<Interface, oneway_t>();
  }

  constexpr bool query(const twoway_t &t) {
    return std::is_same<Interface, twoway_t>();
  }

  Blocking blocking; // # TODO(shrijitsingh99): Fix issue with `static constexpr`
  Interface interface;
};


template <typename Interface, typename Blocking, typename ProtoAllocator>
struct inline_executor: executor<inline_executor, Interface, Blocking, ProtoAllocator> {

  using shape_type = std::size_t;

  template <typename F, typename... Args>
  void execute(F &&f, Args &&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  inline_executor& decay_t() { return *this; };

  inline_executor require(const blocking_t::always_t &t) {
    if constexpr (std::is_same_v<Blocking, blocking_t::always_t>)
      return *this;
    else
      return inline_executor<Interface, blocking_t::always_t, ProtoAllocator>{};
  }

  std::string name() { return "inline"; }
};

template <typename Interface,typename Blocking, typename ProtoAllocator>
struct sse_executor: executor<sse_executor, Interface, Blocking, ProtoAllocator> {

  using shape_type = std::size_t;

  template <typename F, typename... Args>
  void execute(F &&f, Args &&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
  void bulk_execute(F &&f, Args &&... args, std::size_t n) {
    #pragma simd
    for (std::size_t i = 0; i < n; ++i) {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)..., i);
    }
  }

  auto decay_t() -> decltype(auto) {
    if constexpr (execution::executor_available_t<sse_executor>()) {
      return *this;
    }
    else
      return inline_executor<oneway_t, blocking_t::always_t, ProtoAllocator>{};
  }

  std::string name() { return "sse"; }
};

template <typename Interface,typename Blocking, typename ProtoAllocator>
struct omp_executor: executor<sse_executor, Interface, Blocking, ProtoAllocator> {

  using shape_type = std::size_t;

  template <typename F, typename... Args>
  void execute(F &&f, Args &&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
  void bulk_execute(F &&f, shape_type n, Args &&... args) {
    #pragma omp parallel num_threads(n)
    {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)..., 1);
    }
  }

  auto decay_t() -> decltype(auto) {
    if constexpr (execution::executor_available_t<omp_executor>()) {
      return *this;
    }
    else
      return inline_executor<oneway_t, blocking_t::always_t, ProtoAllocator>{};
  }

  std::string name() { return "omp"; }
};



template <typename Interface,typename Blocking, typename ProtoAllocator>
struct cuda_executor: executor<sse_executor, Interface, Blocking, ProtoAllocator> {

  using shape_type = std::vector<std::size_t>;

  template <typename F, typename... Args>
  void bulk_execute(F &&f, shape_type shape, Args &&... args) {
    void *arguments[] = {&args...};
    dim3 g(shape[0], shape[1], shape[2]);
    dim3 b(shape[3], shape[4], shape[5]);
    cudaLaunchKernel((void *)f, g, b, arguments);
    cudaDeviceSynchronize();
  }

  auto decay_t() -> decltype(auto) {
    if constexpr (execution::executor_available_t<cuda_executor>()) {
      return *this;
    }
    else
      return inline_executor<oneway_t, blocking_t::always_t, ProtoAllocator>{};
  }

  std::string name() { return "omp"; }
};