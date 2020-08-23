/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/trait/can_require.hpp>
#include <executor/trait/common_traits.hpp>
#include <type_traits>

namespace executor {

/**
 * \brief Enforces a specified Property on an Executor if possible else returns
 * the same Executor. If enforced a new executor instance which implements that
 * property is created and returned. prefer denotes a customization point and
 * should satisfy the following conditions to be applicable:
 *  1. The Property should be applicable and preferable which can be checked
 * using Property::template is_applicable_property<Executor>::value and
 * Property::is_preferable
 *  2. The expression Property::template static_query<Executor>::value ==
 * Property::value() should be true, which implies that the Executor supports
 * that property
 *
 *  If all the above conditions are met, prefer customization point is valid.
 *  If it is possible to call the require customization point, then it is called
 * and the Property is enforced for the Executor. If the above case is not
 * possible, then the same Executor is returned.
 *
 *  Part of Proposal P1393R0
 *
 * \todo
 * 1. Support multiple querying multiple properties in the trait: template
 * <typename Executor, typename... Properties>
 */
template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property<Executor>::value &&
                  Property::is_preferable && can_require_v<Executor, Property>,
              int> = 0>
constexpr decltype(auto) prefer(const Executor& ex,
                                const Property& p) noexcept {
  return ex.require(p);
}

template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property<Executor>::value &&
                  Property::is_preferable && !can_require_v<Executor, Property>,
              int> = 0>
constexpr decltype(auto) prefer(const Executor& ex,
                                const Property& p) noexcept {
  return ex;
}

/**
 * \brief Checks whether the given Property and Executor support the prefer
 * customization point Part of Proposal P1393R0
 */
template <typename Executor, typename Properties, typename = void>
struct can_prefer : std::false_type {};

template <typename Executor, typename Property>
struct can_prefer<Executor, Property,
                  void_t<decltype(prefer(std::declval<Executor>(),
                                         std::declval<Property>()))>>
    : std::true_type {};

template <typename Executor, typename Property>
constexpr bool can_prefer_v = can_prefer<Executor, Property>::value;

}  // namespace executor
