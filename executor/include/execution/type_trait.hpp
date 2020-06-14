#pragma once

#include <execution/trait/can_require.hpp>
#include <execution/trait/is_executor.hpp>
#include <execution/trait/is_executor_available.hpp>

namespace execution {

template <template <typename...> class Type, typename Executor>
struct is_instance_of : std::false_type {};

template <template <typename...> class Type, typename... Args>
struct is_instance_of<Type, Type<Args...>> : std::true_type {};

template <template <typename...> class Type, typename Executor>
using instance_of =
    std::enable_if_t<is_instance_of<Type, Executor>::value, int>;

}  // namespace execution
