//
// Created by Shrijit Singh on 2020-05-14.
//

#pragma once

#include <experimental/type_traits>
#include <type_traits>
#include <functional>

struct executor {
  virtual std::string name() = 0;
};


template<typename T, typename = void>
struct is_executor : std::false_type {};
template<typename T>
struct is_executor<T, std::enable_if_t<std::is_base_of<executor, T>::value>> : std::true_type {};

template<typename T> inline constexpr bool is_executor_v = is_executor<T>::value;


struct inline_executor_def : executor {};
struct sse_executor_def: executor {};


template <typename T> struct executor_available: std::false_type {};
template <> struct executor_available<inline_executor_def>: std::true_type {};

template <typename T> inline constexpr bool executor_available_v = executor_available<T>::value;

#ifdef SSE_ENABLED
 template <> struct executor_available<sse_executor_def>: std::true_type {};
#endif


struct inline_executor: inline_executor_def {
  template<typename F>
  void execute(F &&f) {
    std::invoke(std::forward<F>(f));
  }
  executor& operator()() { return *this; };

  std::string name() { return "inline"; }
};

struct sse_executor: sse_executor_def {
  executor& operator()() {
    if constexpr (executor_available_v<sse_executor_def>)
      return *this;
    else {
      static inline_executor in_e;
      return in_e;
    }
  }
  std::string name() { return "sse"; }
};