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
 * \brief Checks whether the given executor is available to use
 *
 * \details A given executor is defined as available if the member function
 * `execute` can be called an used without a compile time error. For an executor
 * to be made available it must proved explicit specialization for this trait
 * and inherit from std::true_type
 * E.g. template <> struct is_executor_available<inline_executor> : std::true_type {};
 *
 * This is not part of the proposal and is needed for PCL, since the availability of an executor
 * may depend on hardware availability or other factors
 *
 * \tparam Executor an executor type
 */
template <template <typename...> class Executor>
struct is_executor_available : std::false_type {};

template <template <typename...> class Executor>
static constexpr bool is_executor_available_v = is_executor_available<Executor>::value;

template <typename T>
struct is_executor_instance_available : std::false_type {};

template <template <typename...> class Executor, typename... Properties>
struct is_executor_instance_available<Executor<Properties...>> : is_executor_available<Executor> {};

}  // namespace executor
