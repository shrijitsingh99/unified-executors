/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/trait/common_traits.hpp>
#include <type_traits>

namespace executor {

/**
 * \brief Checks whether a specified Property on an Executor is supported by the
 * Executor. If supported it returns the current instance of that Property.
 *  prefer denotes a customization point and should satisfy the following
 * conditions to be applicable:
 *  1. The Property should be applicable which can be checked using
 * Property::template is_applicable_property<Executor>::value
 *  2. The expression Property::template static_query<Executor>::value should be
 * a valid constant expression
 *
 *  If all the above conditions are met, then the overload query member function
 * in the Executor is called with the Property.
 *
 *  Part of Proposal P1393R0
 *
 */
template <
    typename Executor, typename Property,
    typename std::enable_if_t<
        Property::template is_applicable_property<Executor>::value, int> = 0>
constexpr auto query(Executor&& ex, const Property& p) noexcept {
  return Property::template static_query<Executor>::value;
}

/**
 * \brief Checks whether the given Property and Executor support the query
 * customization point Part of Proposal P1393R0
 */
template <typename Executor, typename Properties, typename = void>
struct can_query : std::false_type {};

template <typename Executor, typename Property>
struct can_query<
    Executor, Property,
    void_t<decltype(query(std::declval<Executor>(), std::declval<Property>()))>>
    : std::true_type {};

template <typename Executor, typename Property>
constexpr bool can_query_v = can_query<Executor, Property>::value;

}  // namespace executor
