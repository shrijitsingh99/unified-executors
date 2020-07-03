//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <executor/default/base_executor.hpp>

template <typename Blocking, typename ProtoAllocator>
struct omp_executor;

#ifdef _OPENMP
namespace executor {
template <>
struct is_executor_available<omp_executor> : std::true_type {};
}  // namespace executor
#endif

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct omp_executor : base_executor<omp_executor, Blocking, ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F>
  void execute(F &&f) const {
    std::forward<F>(f)();
  }

  template <typename F>
  void bulk_execute(F &&f, shape_type n) const {
#ifdef _OPENMP
#pragma omp parallel num_threads(n)
      {std::forward<F>(f)(omp_get_thread_num());
}
#endif
}

omp_executor<blocking_t::always_t, ProtoAllocator> require(
    const blocking_t::always_t &t) const {
  return {};
}

static constexpr auto name() { return "omp"; }
}
;
