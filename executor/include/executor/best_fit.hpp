/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/type_trait.h>

#include <executor/default/cuda_executor.hpp>
#include <executor/default/inline_executor.hpp>
#include <executor/default/omp_executor.hpp>
#include <executor/default/sse_executor.hpp>

namespace executor {

static const auto best_fit_executors =
    std::make_tuple(executor::omp_executor<>{}, executor::sse_executor<>{},
                    executor::inline_executor<>{});

struct executor_runtime_checks {
  template <typename Executor, typename executor::instance_of_base<
                                   Executor, executor::inline_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::instance_of_base<
                                   Executor, executor::sse_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::instance_of_base<
                                   Executor, executor::omp_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::instance_of_base<
                                   Executor, executor::cuda_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }
};

namespace detail {

template <typename Function, typename Executor, typename = void>
bool execute(Function&& f, Executor& exec) {
  return false;
}

template <typename Function, template <typename...> class Executor, typename... Properties,
          typename std::enable_if<executor::is_executor_available_v<Executor>,
                                  int>::type = 0>
bool execute(Function&& f, Executor<Properties...>& exec) {
  f(exec);
  return true;
}

}  // namespace detail

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_with_priority(
    Function f, std::tuple<SupportedExecutors...> supported_execs) {
  static_assert(std::is_base_of<executor_runtime_checks, RuntimeChecks>::value,
                "Runtime checks should inherit from executor_runtime_checks");
  bool executor_selected = false;
  executor::for_each_until_true(supported_execs, [&](auto& exec) {
    if (RuntimeChecks::check(exec)) {
      executor_selected = detail::execute(f, exec);
      return executor_selected;
    }
    return false;
  });
  // TODO: Throw warning or assert
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_with_priority(Function&& f, SupportedExecutors&&... execs) {
  enable_exec_with_priority<RuntimeChecks>(f, std::make_tuple(execs...));
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_on_desc_priority(
    Function f, std::tuple<SupportedExecutors...> supported_execs) {
  static_assert(std::is_base_of<executor_runtime_checks, RuntimeChecks>::value,
                "Runtime checks should inherit from executor_runtime_checks");
  auto num_supported_execs = std::tuple_size<decltype(supported_execs)>::value;
  auto exec_is_supported = [&](auto& best_fit_exec) {
    bool executor_supported = false;
    for_each_until_true(supported_execs, [&](auto& supported_exec) {
      if (executor::is_same_template<decltype(best_fit_exec),
                                     decltype(supported_exec)>::value &&
          RuntimeChecks::check(supported_exec)) {
        executor_supported = detail::execute(f, supported_exec);
        return executor_supported;
      }
      return false;
    });
    return executor_supported;
  };

  bool executor_selected = false;
  for_each_until_true(best_fit_executors, [&](auto& exec) {
    executor_selected = exec_is_supported(exec);
    return executor_selected;
  });
  // TODO: Throw warning or assert
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_on_desc_priority(Function&& f,
                                  SupportedExecutors&&... supported_execs) {
  enable_exec_on_desc_priority<RuntimeChecks>(
      f, std::make_tuple(supported_execs...));
}

}  // namespace executor
