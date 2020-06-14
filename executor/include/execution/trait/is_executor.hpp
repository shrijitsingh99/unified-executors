//
// Created by Shrijit Singh on 14/06/20.
//

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

}  // namespace execution
