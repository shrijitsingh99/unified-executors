/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <type_traits>

namespace executor {

// Part of Standard Library in C++17 onwards

template <typename...>
using void_t = void;

template <typename T>
using remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// In accordance with equality_comparable concept in C++ 20

template <typename T1, typename T2, typename = void>
struct equality_comparable : std::false_type {};

template <typename T1, typename T2>
struct equality_comparable<
    T1, T2,
    executor::void_t<decltype(std::declval<T1>() == std::declval<T2>(),
                              std::declval<T2>() == std::declval<T1>(),
                              std::declval<T1>() != std::declval<T2>(),
                              std::declval<T2>() == std::declval<T1>())>>
    : std::true_type {};

template <typename T1, typename T2>
constexpr bool equality_comparable_v = equality_comparable<T1, T2>::value;

}  // namespace executor
