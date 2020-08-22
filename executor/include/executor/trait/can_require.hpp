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

namespace detail {

/**
 * \brief Checks if the given Executor supports the property P
 * This is checked through a template variable static_query_v which is provided
 * by all properties
 */
template <typename Executor, typename Property>
using contains_property =
    std::is_same<std::remove_const_t<decltype(
                     Property::template static_query<Executor>::value)>,
                 Property>;
}

template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property<Executor>::value &&
                  Property::is_requirable &&
                  detail::contains_property<Executor, Property>::value,
              int> = 0>
constexpr decltype(auto) require(const Executor& ex,
                                 const Property& p) noexcept {
  return ex.require(p);
}

// Part of Proposal P1393R0
/**
 * \brief Checks whether the given Property and Executor support the require customization point
 *  It is used to enforce a specified property on an executor. A new executor instance which implements that property is created and returned.
 *  In future version if the property is already implemented the same instance is returned.
 *
 * \todo
 * 1. Return same instance of executor if property is already implemented
 * 2. Support multiple querying multiple properties in the trait: template <typename Executor, typename... Properties>
 */
template <typename Executor, typename Properties, typename = void>
struct can_require : std::false_type {};

template <typename Executor, typename Property>
struct can_require<Executor, Property,
                   void_t<decltype(require(std::declval<Executor>(),
                                           std::declval<Property>()))>>
    : std::true_type {};

template <typename Executor, typename Property>
constexpr bool can_require_v = can_require<Executor, Property>::value;

}  // namespace executor
