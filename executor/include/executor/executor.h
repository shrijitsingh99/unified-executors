/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <doctest/doctest.h>
#include <executor/property.h>
#include <executor/type_trait.h>

#include <executor/best_fit.hpp>
#include <executor/default/cuda_executor.hpp>
#include <executor/default/inline_executor.hpp>
#include <executor/default/omp_executor.hpp>
#include <executor/default/sse_executor.hpp>

// For printing names of the types for which the test was run
using inline_exec_type =
    executor::inline_executor<executor::blocking_t::always_t>;
using sse_exec_type = executor::sse_executor<executor::blocking_t::always_t>;
using omp_exec_type = executor::omp_executor<executor::blocking_t::always_t>;
using cuda_exec_type = executor::cuda_executor<executor::blocking_t::always_t>;

TYPE_TO_STRING(inline_exec_type);
TYPE_TO_STRING(sse_exec_type);
TYPE_TO_STRING(omp_exec_type);
TYPE_TO_STRING(cuda_exec_type);

// List of Executor types to run tests for
#define TEST_EXECUTORS                                           \
  const inline_exec_type, inline_exec_type, const sse_exec_type, \
      sse_exec_type, const omp_exec_type, omp_exec_type

// Only run certain tests for CUDA executors due to different shape API
// If CUDA is not available fallback to inline executor
#define TEST_CUDA_EXECUTOR                                                     \
  std::conditional<executor::is_executor_available_v<executor::cuda_executor>, \
                   cuda_exec_type, inline_exec_type>::type

TEST_CASE_TEMPLATE_DEFINE("Validity ", E, validity) {
  E exec;

  SUBCASE("is_executor") {
    CHECK(executor::is_executor_v<decltype(exec)> == true);
    CHECK(executor::is_executor_v<int> == false);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Property Traits ", E, property_traits) {
  E exec;

  SUBCASE("can_require") {
    CHECK(executor::can_require_v<E, executor::blocking_t::always_t> == true);
    CHECK(executor::can_require_v<E, executor::blocking_t::never_t> == false);
  }

  SUBCASE("can_prefer") {
    CHECK(executor::can_prefer_v<E, executor::blocking_t::always_t> == true);
    CHECK(executor::can_prefer_v<E, executor::blocking_t::never_t> == true);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Properties", E, properties) {
  E exec;

  SUBCASE("member require & query") {
    CHECK_EQ(exec.query(executor::blocking), executor::blocking_t::always);
    auto new_exec = exec.require(executor::blocking_t::always);
    CHECK_EQ(new_exec.query(executor::blocking), executor::blocking_t::always);
  }

  SUBCASE("function require & query") {
    CHECK_EQ(executor::query(exec, executor::blocking_t::always),
             executor::blocking_t::always);
    auto new_exec = executor::require(exec, executor::blocking_t::always);
    CHECK_EQ(executor::query(new_exec, executor::blocking),
             executor::blocking_t::always);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Execute ", E, execute) {
  E exec;
  int a = 1, b = 2;

  SUBCASE("execute") {
    int c = 0;
    exec.execute([&]() { c = a + b; });
    CHECK(c == 3);
  }

  SUBCASE("bulk_execute") {
    int c[3] = {0};
    exec.bulk_execute(
        [&](auto i) {
          for (auto& val : c) val = 1;
        },
        3);
    CHECK(c[0] + c[1] + c[2] == 3);
  }
}

TEST_CASE_TEMPLATE_APPLY(validity,
                         std::tuple<TEST_EXECUTORS, TEST_CUDA_EXECUTOR>);
TEST_CASE_TEMPLATE_APPLY(property_traits,
                         std::tuple<TEST_EXECUTORS, TEST_CUDA_EXECUTOR>);
TEST_CASE_TEMPLATE_APPLY(properties,
                         std::tuple<TEST_EXECUTORS, TEST_CUDA_EXECUTOR>);
TEST_CASE_TEMPLATE_APPLY(execute, std::tuple<TEST_EXECUTORS>);
