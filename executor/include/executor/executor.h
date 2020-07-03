#pragma once

#include <doctest/doctest.h>

#include <executor/default/cuda_executor.hpp>
#include <executor/default/inline_executor.hpp>
#include <executor/default/omp_executor.hpp>
#include <executor/default/sse_executor.hpp>
#include <executor/trait/can_prefer.hpp>
#include <executor/trait/can_query.hpp>
#include <executor/trait/can_require.hpp>

// For printing names of the types for which the test was run
TYPE_TO_STRING(executor::inline_executor<executor::blocking_t::always_t>);
TYPE_TO_STRING(executor::sse_executor<executor::blocking_t::always_t>);
TYPE_TO_STRING(executor::omp_executor<executor::blocking_t::always_t>);
TYPE_TO_STRING(executor::cuda_executor<executor::blocking_t::always_t>);

// List of Executor types to run tests for
#define TEST_EXECUTORS                                              \
  const executor::inline_executor<executor::blocking_t::always_t>,  \
      const executor::sse_executor<executor::blocking_t::always_t>, \
      const executor::omp_executor<executor::blocking_t::always_t>

// Only run certain tests for CUDA executors due to different shape API
// If CUDA is not available fallback to inline executor
#define TEST_CUDA_EXECUTOR                                        \
  std::conditional<                                               \
      executor::is_executor_available_v<executor::cuda_executor>, \
      executor::cuda_executor<executor::blocking_t::always_t>,    \
      executor::inline_executor<executor::blocking_t::always_t>>::type

TEST_CASE_TEMPLATE_DEFINE("Validity ", E, validity) {
  auto exec = E{};

  SUBCASE("is_executor") {
    CHECK(executor::is_executor_v<decltype(exec)> == true);
    CHECK(executor::is_executor_v<int> == false);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Property Traits ", E, property_traits) {
  auto exec = E{};

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
  auto exec = E{};

  SUBCASE("member require & query") {
    CHECK(exec.query(blocking.never) == false);
    auto new_exec = exec.require(blocking.always);
    CHECK(new_exec.query(blocking.always) == true);
  }

  SUBCASE("function require & query") {
    CHECK(execution::query(exec, blocking.never) == false);
    auto new_exec = exec.require(blocking.always);
    CHECK(execution::query(new_exec, blocking.always) == true);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Execute ", E, execute) {
  auto exec = E{};
  int a = 1, b = 2;

  SUBCASE("execute") {
    int c = 0;
    exec.execute([&]() { c = a + b; });
    CHECK(c == 3);
  }

  SUBCASE("bulk_execute") {
    int c[3] = {0};
    exec.bulk_execute([&](int i) { c[i] = 1; }, 3);
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
