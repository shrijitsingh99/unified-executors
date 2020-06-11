#pragma once

#include <experimental/type_traits>
#include <type_traits>

namespace execution {
template <class Executor, typename = void>
struct is_executor : std::false_type {};

constexpr const auto noop = [] {};

template <class Executor>
struct is_executor<
    Executor,
    std::enable_if_t<
        std::is_same_v<void, decltype(std::declval<Executor>().execute(noop))>,
        void>> : std::true_type {};

template <typename Executor>
using is_executor_t = typename is_executor<Executor>::type;

template <template <typename, typename, typename, typename> typename T>
struct executor_available : std::false_type {};
template <template <typename, typename, typename, typename> typename T>
using executor_available_t = typename executor_available<T>::type;

template <template <typename...> class Type, typename Executor>
struct is_executor_of_type : std::false_type {};

template <template <typename...> class Type, typename... Args>
struct is_executor_of_type<Type, Type<Args...>> : std::true_type {};

template <template <typename...> class Type, typename Executor>
using executor_of_type =
std::enable_if_t<is_executor_of_type<Type, Executor>::value, int>;

} // namespace execution