#pragma once

#include <sfinae_traits.hpp>
#include <sfinae_prop.hpp>
#include <functional>
#include <string>
#include <iostream>


#define SSE


struct inline_executor_def : executor {};
struct sse_executor_def: executor {};

template <typename T> struct executor_available: std::false_type {};
template <> struct executor_available<inline_executor_def>: std::true_type {};

template <typename T> inline constexpr bool executor_available_v = executor_available<T>::value;

#ifdef SSE
template <> struct executor_available<sse_executor_def>: std::true_type {};
#endif


struct inline_executor: inline_executor_def {
  template<typename F, typename ... Args>
  auto execute(F&& f, Args&&... args) -> decltype(auto){
    return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  inline_executor& fallback_t() { return *this; };

  using shape_type = std::size_t;

  shape_type shape;

  inline_executor require(blocking_t::always_t) {
    return *this;
  }

  std::string name() { return "inline"; }
};

  struct sse_executor: sse_executor_def, inline_executor {
  auto fallback_t() -> decltype(auto) {
    if constexpr (executor_available_v<sse_executor_def>)
      return *this;
    else
      return inline_executor{};
  }

  std::string name() { return "sse"; }
};
