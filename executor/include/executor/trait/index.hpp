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
 * \brief A given Executor can have a custom index for bulk execute.
 *
 * \details: By default if not explicitly specified by the executor the shape is std::size_t.
 * The index represents the index of execution unit which is currently running.
 *
 * executor_index is an Executor type trait which provides the index type
 * defined by the Executor.
 *
 * Any executor can define a custom index by defining the alias index_type for
 * the custom index.
 *
 * Part of proposal P0443R13
 */
template <typename Executor, typename = void>
struct executor_index {
  using type = std::size_t;
};  // namespace executor

template <typename Executor>
struct executor_index<Executor,
                      executor::void_t<typename Executor::index_type>> {
  using type = typename Executor::index_type;
};

template <class Executor>
using executor_index_t = typename executor_index<Executor>::type;

}  // namespace executor
