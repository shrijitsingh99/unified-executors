/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

namespace executor {

/**
 * \brief A given Executor can have a custom shape for bulk execute.
 *
 * \details By default if not explicitly specified by the executor the shape is std::size_t.
 * The shape represents the number of execution units and their dimensionality if
 * needed.
 *
 * executor_shape is an Executor type trait which provides the shape type
 * defined by the Executor.
 *
 * Any executor can define a custom shape by defining the alias shape_type for
 * the custom shape.
 *
 * Part of proposal P0443R13
 */
template <typename Executor, typename = void>
struct executor_shape {
  using type = std::size_t;
};

template <typename Executor>
struct executor_shape<Executor,
                      executor::void_t<typename Executor::shape_type>> {
  using type = typename Executor::shape_type;
};

template <class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;

}  // namespace executor
