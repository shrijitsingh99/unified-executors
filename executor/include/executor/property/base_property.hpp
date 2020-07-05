/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/trait/is_executor.hpp>
#include <functional>
#include <iostream>
#include <string>

namespace executor {

template <typename Derived, bool requireable, bool preferable>
struct basic_executor_property {
  static constexpr bool is_requirable = requireable;
  static constexpr bool is_preferable = preferable;

  template <class T>
  static constexpr bool is_applicable_property() {
    return executor::is_executor<T>();
  }

  template <class Executor>
  static constexpr auto static_query() {
    return executor::remove_cv_ref_t<Executor>::query(Derived{});
  }

  template <typename T>
  static constexpr bool is_applicable_property_v = is_applicable_property<T>();

  template <class Executor>
  static constexpr decltype(auto) static_query_v = static_query<Executor>();
};

}  // namespace executor
