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

  template <typename F, typename... Args>
  void execute(F &&f, Args &&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
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

TEST_CASE("inline_executor execute") {
  auto exec = inline_executor<oneway_t, single_t, blocking_t::always_t, void>{};
  int a = 1, b = 2, c = 0;
  exec.execute([&]() { c = a + b; });
  CHECK(c == 3);
}