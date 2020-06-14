//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <execution/executor/base_executor.hpp>

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator>
struct omp_executor;

#ifdef _OPENMP
template <>
struct execution::is_executor_available<omp_executor> : std::true_type {};
#endif

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator>
struct omp_executor
    : executor<omp_executor, Interface, Cardinality, Blocking, ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F, typename... Args> void execute(F &&f, Args &&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
  void bulk_execute(F &&f, shape_type n, Args &&... args) {
#pragma omp parallel num_threads(n)
    { std::invoke(std::forward<F>(f), std::forward<Args>(args)..., 1); }
  }

  auto decay_t() -> decltype(auto) {
    if constexpr (execution::is_executor_available_t<omp_executor>()) {
      return *this;
    } else
      return inline_executor<oneway_t, single_t, blocking_t::always_t,
                             ProtoAllocator>{};
  }

  std::string name() { return "omp"; }
};
