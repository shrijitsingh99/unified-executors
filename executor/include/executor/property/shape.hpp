/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/property/base_property.hpp>

namespace executor {

template <class Executor>
struct executor_shape
    : basic_executor_property<executor_shape<Executor>, true, true> {
  // private:
  //  template <class T> using helper = typename T::shape_type;

 public:
  template <unsigned _s0 = 0, unsigned... _sizes>
  using type = std::size_t;
  //  using type = std::experimental::detected_or_t<size_t, helper, Executor>;
  //
  //  static_assert(std::is_integral_v<type>,
  //                "shape type must be an integral type");
};

template <class Executor>
struct executor_shape;

template <class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;

}  // namespace executor
