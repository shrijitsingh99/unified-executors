/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
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
  template <typename Executor, typename executor::InstanceOf<
                                   Executor, executor::inline_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::InstanceOf<
                                   Executor, executor::sse_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::InstanceOf<
                                   Executor, executor::omp_executor> = 0>
  static bool check(Executor& exec) {
    return true;
  }

  template <typename Executor, typename executor::InstanceOf<
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

template <typename Supported>
struct executor_predicate {
  template <typename T, typename = void>
  struct condition: std::false_type{};

  template<typename T>
  struct condition<T, std::enable_if_t<is_executor_instance_available<T>::value && tuple_contains_type<T, Supported>::value>>: std::true_type{};
};

}  // namespace detail

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_with_priority(
    Function&& f, std::tuple<SupportedExecutors...> supported_execs) {
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


  // This should not happen, at least one executor should always be selected. So either all runtime checks failed which is
  // incorrect behaviour or no supported executors were passed which should not be done
  if (!executor_selected)
      std::cerr<<"No executor selected. All runtime checks returned false or no executors were passed."<<std::endl;
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_with_priority(Function&& f, SupportedExecutors&&... execs) {
  enable_exec_with_priority<RuntimeChecks>(f, std::make_tuple(execs...));
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_on_desc_priority(
    Function&& f, std::tuple<SupportedExecutors...> supported_execs) {
  static_assert(std::is_base_of<executor_runtime_checks, RuntimeChecks>::value,
                "Runtime checks should inherit from executor_runtime_checks");

  using predicate = detail::executor_predicate<decltype(supported_execs)>;
  filter_tuple_values<predicate::template condition, decltype(best_fit_executors)> filter_available;

  auto filtered = filter_available(best_fit_executors);

  enable_exec_with_priority(f, filtered);
}

template <typename RuntimeChecks = executor_runtime_checks, typename Function,
          typename... SupportedExecutors>
void enable_exec_on_desc_priority(Function&& f,
                                  SupportedExecutors&&... supported_execs) {
  enable_exec_on_desc_priority<RuntimeChecks>(
      f, std::make_tuple(supported_execs...));
}

}  // namespace executor
