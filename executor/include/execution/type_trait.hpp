#pragma once

#include <execution/trait/is_executor.hpp>
#include <execution/trait/is_executor_available.hpp>

namespace execution {

template <template <typename...> class Type, typename Executor>
struct is_executor_of_type : std::false_type {};

template <template <typename...> class Type, typename... Args>
struct is_executor_of_type<Type, Type<Args...>> : std::true_type {};

template <template <typename...> class Type, typename Executor>
using executor_of_type =
    std::enable_if_t<is_executor_of_type<Type, Executor>::value, int>;

}  // namespace execution