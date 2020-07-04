/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <executor/default/base_executor.hpp>

namespace executor {

template <typename Blocking, typename ProtoAllocator>
struct omp_executor;

#ifdef _OPENMP
template <>
struct is_executor_available<omp_executor> : std::true_type {};
#endif

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct omp_executor : base_executor<omp_executor, Blocking, ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F>
  void execute(F&& f) const {
    std::forward<F>(f)();
  }

  template <typename F>
  void bulk_execute(F&& f, shape_type n) const {
#ifdef _OPENMP
  #pragma omp parallel for num_threads(n)
    for (int i = 0; i < n; ++i) std::forward<F>(f)(omp_get_thread_num());
#endif
  }

  omp_executor<blocking_t::always_t, ProtoAllocator> require(
      const blocking_t::always_t& t) const {
    return {};
  }

  static constexpr auto name() { return "omp"; }
};

}  // namespace executor
