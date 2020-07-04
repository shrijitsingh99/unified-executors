/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/default/base_executor.hpp>

namespace executor {

template <typename Blocking, typename ProtoAllocator>
struct inline_executor;

template <>
struct is_executor_available<inline_executor> : std::true_type {};

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct inline_executor
    : base_executor<inline_executor, Blocking, ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F>
  void execute(F&& f) const {
    std::forward<F>(f)();
  }

  inline_executor<blocking_t::always_t, ProtoAllocator> require(
      const blocking_t::always_t& t) const {
    return {};
  }

  static constexpr auto name() { return "inline"; }
};

}  // namespace executor
