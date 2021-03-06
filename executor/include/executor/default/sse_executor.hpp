/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/property.h>
#include <executor/type_trait.h>

namespace executor {

template <typename Blocking, typename ProtoAllocator>
struct sse_executor;

#ifdef __SSE__
template <>
struct is_executor_available<sse_executor> : std::true_type {};
#endif

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct sse_executor {
  using shape_type = std::size_t;

  template <typename Executor, InstanceOf<Executor, sse_executor> = 0>
  friend bool operator==(const sse_executor& lhs,
                         const Executor& rhs) noexcept {
    return std::is_same<sse_executor, Executor>::value;
  }

  template <typename Executor, InstanceOf<Executor, sse_executor> = 0>
  friend bool operator!=(const sse_executor& lhs,
                         const Executor& rhs) noexcept {
    return !operator==(lhs, rhs);
  }

  template <typename F>
  void execute(F&& f) const {
    static_assert(is_executor_available_v<sse_executor>,
                  "SSE executor unavailable");
    f();
  }

  template <typename F>
  void bulk_execute(F&& f, const shape_type& n) const {
    static_assert(is_executor_available_v<sse_executor>,
                  "SSE executor unavailable");
    // TODO: Look into what bulk execute will do for SSE
#pragma simd
    f(0);
  }

  static constexpr auto query(const blocking_t&) noexcept { return Blocking{}; }

  sse_executor<blocking_t::always_t, ProtoAllocator> require(
      const blocking_t::always_t&) const {
    return {};
  }

  static constexpr auto name() { return "sse_executor"; }
};

using default_sse_executor = sse_executor<>;

}  // namespace executor
