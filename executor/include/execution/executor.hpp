#pragma once

#include <doctest/doctest.h>

#include <execution/executor/base_executor.hpp>
#include <execution/executor/cuda_executor.hpp>
#include <execution/executor/inline_executor.hpp>
#include <execution/executor/omp_executor.hpp>
#include <execution/executor/sse_executor.hpp>

TYPE_TO_STRING(inline_executor<oneway_t, single_t, blocking_t::always_t>);
TYPE_TO_STRING(sse_executor<oneway_t, bulk_t, blocking_t::always_t>);
TYPE_TO_STRING(omp_executor<oneway_t, bulk_t, blocking_t::always_t>);
TYPE_TO_STRING(cuda_executor<oneway_t, bulk_t, blocking_t::always_t>);

#define executors                                                \
  std::tuple<                                                    \
      inline_executor<oneway_t, single_t, blocking_t::always_t>, \
      sse_executor<oneway_t, bulk_t, blocking_t::always_t>,      \
      omp_executor<oneway_t, bulk_t, blocking_t::always_t>,      \
      std::conditional<                                          \
          execution::is_executor_available_v<cuda_executor>,     \
          cuda_executor<oneway_t, bulk_t, blocking_t::always_t>, \
          inline_executor<oneway_t, single_t, blocking_t::always_t>>::type>

TEST_CASE_TEMPLATE_DEFINE("Validity ", E, validity) {
  auto exec = E{};

  SUBCASE("is_executor") {
    CHECK(execution::is_executor_v<decltype(exec)> == true);
    CHECK(execution::is_executor_v<int> == false);
  }
}

TEST_CASE_TEMPLATE_DEFINE("Property Traits ", E, property_traits) {
  auto exec = E{};

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
  auto exec = inline_executor<oneway_t, single_t, blocking_t::never_t>{};

  SUBCASE("require & query") {
    CHECK(exec.query(blocking.never));
    auto new_exec = exec.require(blocking.always);
    CHECK(new_exec.query(blocking.always));
  }
}

TEST_CASE_TEMPLATE_DEFINE("Execute ", E, execute) {
  auto exec = E{};
  int a = 1, b = 2, c = 0;

  SUBCASE("execute") {
    c = 0;
    exec.execute([&]() { c = a + b; });
    CHECK(c == 3);
  }

  SUBCASE("bulk_execute") {
    int c[3] = {0};
    exec.bulk_execute([&](int i) { c[i] = 1; }, 3);
    CHECK(c[0] + c[1] + c[2] == 3);
  }
}

TEST_CASE_TEMPLATE_APPLY(validity, executors);
TEST_CASE_TEMPLATE_APPLY(property_traits, executors);
TEST_CASE_TEMPLATE_APPLY(properties, executors);
TEST_CASE_TEMPLATE_APPLY(execute, executors);
