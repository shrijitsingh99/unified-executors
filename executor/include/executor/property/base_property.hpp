/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/trait/is_executor.hpp>
#include <functional>

namespace executor {

template <typename Derived, bool requireable, bool preferable>
struct basic_executor_property {
  // Part of Proposal P1393R0
  static constexpr bool is_requirable = requireable;
  static constexpr bool is_preferable = preferable;

  // Part of Proposal P1393R0
  template <class Executor>
  struct is_applicable_property {
    static constexpr bool value = executor::is_executor<Executor>();
  };

  // Part of Proposal P0443R13: 2.2.11 & 2.2.12
  template <class Executor>
  struct static_query {
    static constexpr auto value =
        std::remove_reference_t<Executor>::query(Derived{});
  };

  template <typename T>
  static constexpr bool is_applicable_property_v =
      is_applicable_property<T>::value;

  template <class Executor>
  static constexpr auto static_query_v = static_query<Executor>::value;
};

}  // namespace executor
