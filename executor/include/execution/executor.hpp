#pragma once

#include <doctest/doctest.h>

#include <execution/executor/cuda_executor.hpp>
#include <execution/executor/inline_executor.hpp>
#include <execution/executor/omp_executor.hpp>
#include <execution/executor/sse_executor.hpp>
#include <execution/trait/can_prefer.hpp>
#include <execution/trait/can_query.hpp>
#include <execution/trait/can_require.hpp>

// For printing names of the types for which the test was run
TYPE_TO_STRING(inline_executor<oneway_t, single_t, blocking_t::always_t>);
TYPE_TO_STRING(sse_executor<oneway_t, bulk_t, blocking_t::always_t>);
TYPE_TO_STRING(omp_executor<oneway_t, bulk_t, blocking_t::always_t>);
TYPE_TO_STRING(cuda_executor<oneway_t, bulk_t, blocking_t::always_t>);

// List of Executor types to run tests for
#define TEST_EXECUTORS                                       \
  inline_executor<oneway_t, single_t, blocking_t::always_t>, \
      sse_executor<oneway_t, bulk_t, blocking_t::always_t>,  \
      omp_executor<oneway_t, bulk_t, blocking_t::always_t>

// Only run certain tests for CUDA executors due to different shape API
// If CUDA is not available fallback to inline executor
#define TEST_CUDA_EXECUTOR                                   \
  std::conditional<                                          \
      execution::is_executor_available_v<cuda_executor>,     \
      cuda_executor<oneway_t, bulk_t, blocking_t::always_t>, \
      inline_executor<oneway_t, single_t, blocking_t::always_t>>::type

TEST_CASE_TEMPLATE_DEFINE("Validity ", E, validity) {
  const auto exec = E{};

  SUBCASE("is_executor") {
    CHECK(execution::is_executor_v<decltype(exec)> == true);
    CHECK(execution::is_executor_v<int> == false);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Property Traits ", E, property_traits) {
  const auto exec = E{};

  SUBCASE("can_require") {
    CHECK(execution::can_require_v<decltype(exec), blocking_t::always_t> ==
          true);
    CHECK(execution::can_require_v<decltype(exec), blocking_t::never_t> ==
          false);
  }

  SUBCASE("can_prefer") {
    CHECK(execution::can_prefer_v<decltype(exec), blocking_t::always_t> ==
          true);
    CHECK(execution::can_prefer_v<decltype(exec), blocking_t::never_t> ==
          false);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Properties", E, properties) {
  const auto exec = E{};

  SUBCASE("member require & query") {
    CHECK(exec.query(blocking.never) == false);
    auto new_exec =
        static_cast<E>(exec).require(oneway).require(blocking.always);
    CHECK((new_exec.query(oneway) && new_exec.query(blocking.always)) == true);
  }

  SUBCASE("function require & query") {
    CHECK(execution::query(exec, blocking.never) == false);
    auto new_exec =
        static_cast<E>(exec).require(oneway).require(blocking.always);
    CHECK((execution::query(new_exec, oneway) &&
           execution::query(new_exec, blocking.always)) == true);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Execute ", E, execute) {
  const auto exec = E{};
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
