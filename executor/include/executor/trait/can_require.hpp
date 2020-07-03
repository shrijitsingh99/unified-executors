//
// Created by Shrijit Singh on 14/06/20.
//

#pragma once

#include <executor/trait/common_traits.hpp>
#include <type_traits>

namespace executor {

template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property_v<Executor> &&
                  Property::is_requirable &&
                  Property::template static_query<Executor>(),
              int> = 0>
constexpr auto require(Executor &&ex, const Property &p) noexcept {
  return ex.require(p);
}

template <typename Executor, typename Properties, typename = void>
struct can_require : std::false_type {};

template <typename Executor, typename Property>
struct can_require<Executor, Property,
                   void_t<decltype(require(
                       std::declval<execution::remove_cv_ref_t<Executor>>(),
                       std::declval<execution::remove_cv_ref_t<Property>>()))>>
    : std::true_type {};

template <typename Executor, typename Property>
constexpr bool can_require_v = can_require<Executor, Property>::value;

}  // namespace executor
