//
// Created by Shrijit Singh on 17/06/20.
//

#pragma once

#include <execution/trait/can_require.hpp>
#include <execution/trait/common_traits.hpp>
#include <type_traits>

namespace execution {

template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property_v<Executor> &&
                  Property::is_preferable,
              int> = 0>
constexpr auto prefer(Executor &&ex, const Property &t) noexcept {
  if (execution::can_require_v<Executor, Property>)
    return std::forward<Executor>(ex).require(t);
  return std::forward<Executor>(ex);
}

template <typename Executor, typename Properties, typename = void>
struct can_prefer : std::false_type {};

template <typename Executor, typename Property>
struct can_prefer<Executor, Property,
                  COMMON_TRAIT_NS::void_t<decltype(execution::require(
                      std::declval<Executor>(), std::declval<Property>()))>>
    : std::true_type {};

template <typename Executor, typename Property>
constexpr bool can_prefer_v = can_prefer<Executor, Property>::value;

}  // namespace execution
