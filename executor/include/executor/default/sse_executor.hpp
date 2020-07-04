//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <executor/default/base_executor.hpp>

namespace executor {

template <typename Blocking, typename ProtoAllocator>
struct sse_executor;

#ifdef __SSE__
template <>
struct is_executor_available<sse_executor> : std::true_type {};
#endif

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct sse_executor : base_executor<sse_executor, Blocking, ProtoAllocator> {
  using shape_type = std::size_t;

  template <typename F>
  void execute(F&& f) const {
    std::forward<F>(f)();
  }

  template <typename F>
  void bulk_execute(F&& f, shape_type n) const {
#pragma simd
    for (std::size_t i = 0; i < n; ++i) {
      std::forward<F>(f)(i);
    }
  }

  sse_executor<blocking_t::always_t, ProtoAllocator> require(
      const blocking_t::always_t& t) const {
    return {};
  }

  static constexpr auto name() { return "sse"; }
};

}  // namespace executor
