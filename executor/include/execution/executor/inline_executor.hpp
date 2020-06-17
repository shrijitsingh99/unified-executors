//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <execution/executor/base_executor.hpp>

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator>
struct inline_executor;

template <>
struct execution::is_executor_available<inline_executor> : std::true_type {};

template <typename Interface, typename Cardinality, typename Blocking,
          typename ProtoAllocator = std::allocator<void>>
struct inline_executor
    : public executor<inline_executor, Interface, Cardinality, Blocking,
                      ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F>
  void execute(F &&f) {
    std::invoke(std::forward<F>(f));
  }

  inline_executor &decay_t() { return *this; };

  inline_executor require(const blocking_t::always_t &t) {
    if constexpr (std::is_same_v<Blocking, blocking_t::always_t>)
      return *this;
    else
      return inline_executor<Interface, blocking_t::always_t, ProtoAllocator>{};
  }

  std::string name() { return "inline"; }
};

TEST_CASE("inline_executor Validity") {
  auto exec = inline_executor<oneway_t, single_t, blocking_t::always_t>{};

  SUBCASE("inline_executor is_executor_avilable") {
    CHECK(execution::is_executor_available_v<inline_executor> == true);
  }

  SUBCASE("inline_executor is_executor") {
    CHECK(execution::is_executor_v<decltype(exec)> == true);
  }

  SUBCASE("inline_executor is_instance_of_base") {
    CHECK(execution::is_instance_of_base_v<inline_executor, decltype(exec)> ==
          true);
  }
}

TEST_CASE("inline_executor Properties") {
  auto exec = inline_executor<oneway_t, single_t, blocking_t::always_t>{};

  SUBCASE("inline_executor can_require") {
    CHECK(execution::can_require_v<decltype(exec), blocking_t::always_t> ==
          true);
    CHECK(execution::can_require_v<decltype(exec), blocking_t::never_t> ==
          false);
  }

  SUBCASE("inline_executor can_prefer") {
    CHECK(execution::can_prefer_v<decltype(exec), blocking_t::always_t> ==
          true);
    CHECK(execution::can_prefer_v<decltype(exec), blocking_t::never_t> ==
          false);
  }
}

TEST_CASE("inline_executor Execute") {
  auto exec = inline_executor<oneway_t, single_t, blocking_t::always_t>{};
  int a = 1, b = 2, c = 0;

  SUBCASE("inline_executor execute") {
    c = 0;
    exec.execute([&]() { c = a + b; });
    CHECK(c == 3);
  }

  SUBCASE("inline_executor bulk_execute") {
    c = 0;
    exec.bulk_execute([&](int i) { c += i; }, 3);
    CHECK(c == 3);
  }
}
