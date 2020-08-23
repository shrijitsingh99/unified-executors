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
#include <experimental/type_traits>
#include <type_traits>

namespace executor {

namespace detail {
/**
 * \todo Convert to a static constexpr in C++ 17 onwards
 * Lambda closure types are non-literal types before C++17
 */
static const auto noop = [] {};

/**
 * \brief Checks if the given type provides a member function named `execute` that takes a
 * nullary callable
 * \tparam T type to check for member function `execute`
 */
template <typename T, typename = void>
struct contains_execute : std::false_type {};

template <typename T>
struct contains_execute<
    T, std::enable_if_t<std::is_same<
           decltype(std::declval<T>().execute(detail::noop)), void>::value>>
    : std::true_type {};

}  // namespace detail

/**
 * \brief Checks whether the type is an executor
 *
 * \details A given type T is an Executor if it satisfies the following
 * properties:
 *  1. Provides a function named execute that eagerly submits work on a single
 * execution agent created for it by the executor
 *  2. Is a CopyConstructible type
 *  3. Is a EqualityComparable type
 *
 *  This concept was finalized in P1660R0 and merged in the draft P0443R13 in
 * the form of a concept
 *
 * \tparam T type to check whether is an executor or not
 *  \todo Convert to a concept in C++20
 */
template <class T, typename = void>
struct is_executor : std::false_type {};

template <typename T>
struct is_executor<T,
                   std::enable_if_t<std::is_copy_constructible<T>::value &&
                                    detail::contains_execute<T>::value &&
                                    executor::equality_comparable<T, T>::value>>
    : std::true_type {};

template <typename T>
static constexpr bool is_executor_v = is_executor<T>::value;

}  // namespace executor
