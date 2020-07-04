//
// Created by Shrijit Singh on 14/06/20.
//

#pragma once

#include <executor/default/base_executor.hpp>
#include <executor/trait/common_traits.hpp>
#include <experimental/type_traits>
#include <type_traits>

namespace executor {

template <class Executor, typename = void>
struct is_executor : std::false_type {};

namespace detail {
const auto noop = [] {};
}

template <typename Executor>
struct is_executor<
    Executor,
    std::enable_if_t<
        std::is_copy_constructible<
            executor::remove_cv_ref_t<Executor>>::value &&
            std::is_same<decltype(
                             std::declval<executor::remove_cv_ref_t<Executor>>()
                                 .execute(detail::noop)),
                         void>::value,
        void_t<decltype(std::declval<Executor>() == std::declval<Executor>())>>>
    : std::true_type {};

template <typename Executor>
constexpr bool is_executor_v = is_executor<Executor>::value;

}  // namespace executor
