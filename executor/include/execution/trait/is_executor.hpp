//
// Created by Shrijit Singh on 14/06/20.
//

#pragma once

#include <execution/executor/base_executor.hpp>
#include <experimental/type_traits>
#include <type_traits>

namespace execution {

template <class Executor, typename = void>
struct is_executor : std::false_type {};

constexpr const auto noop = [] {};

template <class Executor>
struct is_executor<Executor, std::enable_if_t<true, void>> : std::true_type {};

// #TODO: Satisfy executor conditions
// (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1660r0.pdf)

template <typename Executor>
using is_executor_t = typename is_executor<Executor>::type;

}  // namespace execution
