#pragma once

#include <execution/property.hpp>
#include <execution/type_traits.hpp>
#include <functional>
#include <iostream>
#include <string>

#define SSE


template <typename T> struct executor_available : std::false_type {};
template <typename T> using executor_available_t = typename executor_available<T>::type;


template <typename...> struct inline_executor;
template <> struct executor_available<inline_executor<>> : std::true_type {};

template <typename...> struct sse_executor;
#ifdef SSE
template <> struct executor_available<sse_executor<>> : std::true_type {};
#endif

template <typename Blocking> struct inline_executor<Blocking> {
  template <typename F, typename... Args>
  auto execute(F &&f, Args &&... args) -> decltype(auto) {
    return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }

  inline_executor& decayt_t() { return *this; };

  using shape_type = std::size_t;

  shape_type shape;

  inline_executor require(const blocking_t::always_t &t) {
    if constexpr (std::is_same_v<Blocking, blocking_t::always_t>)
      return *this;
    else
      return inline_executor<blocking_t::always_t>{};
  }

  std::string name() { return "inline"; }
};

template <typename Blocking>
struct sse_executor<Blocking> : inline_executor<Blocking> {
  auto decay_t() -> decltype(auto) {
    if constexpr (executor_available_t<sse_executor<>>()) {
      return *this;
    }
    else
      return inline_executor<blocking_t::always_t>{};
  }

  std::string name() { return "sse"; }
};
