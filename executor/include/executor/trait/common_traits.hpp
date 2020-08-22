/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <tuple>
#include <type_traits>

namespace executor {

/**
 *   Part of Standard Library in C++17 onwards
 **/

// void_t
template <typename...>
using void_t = void;

// remove_cv_ref_t
template <typename T>
using remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// disjunction
// https://stackoverflow.com/questions/31533469/check-a-parameter-pack-for-all-of-type-t
template <typename... Conds>
struct disjunction : std::false_type {};

template <typename Cond, typename... Conds>
struct disjunction<Cond, Conds...>
    : std::conditional<Cond::value, std::true_type,
                       disjunction<Conds...>>::type {};

/**
 *   In accordance with equality_comparable concept in C++ 20
 **/

template <typename T1, typename T2, typename = void>
struct equality_comparable : std::false_type {};

template <typename T1, typename T2>
struct equality_comparable<
    T1, T2,
    executor::void_t<decltype(std::declval<T1>() == std::declval<T2>(),
                              std::declval<T2>() == std::declval<T1>(),
                              std::declval<T1>() != std::declval<T2>(),
                              std::declval<T2>() != std::declval<T1>())>>
    : std::true_type {};

template <typename T1, typename T2>
constexpr bool equality_comparable_v = equality_comparable<T1, T2>::value;

/**
 *   Custom Traits
 **/

// is_instance_of_base
namespace detail {

template <typename Executor, template <typename...> class Type, typename = void>
struct is_instance_of_any_impl : std::false_type {};

template <template <typename...> class Executor,
          template <typename...> class Type, typename... Args>
struct is_instance_of_any_impl<
    Executor<Args...>, Type,
    std::enable_if_t<std::is_base_of<Type<Args...>, Executor<Args...>>::value>>
    : std::true_type {};

}  // namespace detail

template <typename Executor, template <typename...> class... Type>
using is_instance_of_any =
    executor::disjunction<detail::is_instance_of_any_impl<Executor, Type>...>;

template <typename Executor, template <typename...> class... Type>
constexpr bool is_instance_of_any_v =
    is_instance_of_any<Executor, Type...>::value;

template <typename Executor, template <typename...> class... Type>
using InstanceOfAny =
    std::enable_if_t<is_instance_of_any_v<Executor, Type...>, int>;

template <typename Executor, template <typename...> class Type>
using InstanceOf =
std::enable_if_t<is_instance_of_any_v<Executor, Type>, int>;

// is_same_template
namespace detail {

template <typename T1, typename T2>
struct is_same_template_impl : std::false_type {};

template <template <typename...> class Type, typename... Args1,
          typename... Args2>
struct is_same_template_impl<Type<Args1...>, Type<Args2...>> : std::true_type {
};

}  // namespace detail

template <typename T1, typename T2>
using is_same_template =
    detail::is_same_template_impl<executor::remove_cv_ref_t<T1>,
                                  executor::remove_cv_ref_t<T2>>;

// for_each_tuple_until
// Iterate over tuple
// https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11
template <typename TupleType, typename FunctionType>
void for_each_until_true(
    TupleType&&, FunctionType,
    std::integral_constant<std::size_t,
                           std::tuple_size<typename std::remove_reference<
                               TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
void for_each_until_true(TupleType&& t, FunctionType f,
                          std::integral_constant<size_t, I>) {
  bool exit = f(std::get<I>(std::forward<TupleType>(t)));

  if (!exit)
    for_each_until_true(std::forward<TupleType>(t), f,
                         std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void for_each_until_true(TupleType&& t, FunctionType f) {
  for_each_until_true(std::forward<TupleType>(t), f,
                       std::integral_constant<size_t, 0>());
}


// tuple_contains_type
namespace detail {

template <typename T, typename Tuple>
struct tuple_contains_type_impl;

template <typename T, typename... Us>
struct tuple_contains_type_impl<T, std::tuple<Us...>>
    : disjunction<std::is_same<T, Us>...> {};

}  // namespace detail

template <typename T, typename Tuple>
using tuple_contains_type = typename detail::tuple_contains_type_impl<T, Tuple>::type;


// filter_tuple_values
namespace detail {

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values_impl {
  using type = decltype(
  std::tuple_cat(typename std::conditional<predicate<T>::value, std::tuple<T>,
                                           std::tuple<>>::type()...));

  auto operator()(const std::tuple<T...> &in) { return (*this)(in, type{}); }

 private:
  // neat utility function to fetch the types we're interest in outputting
  template<typename... To>
  auto operator()(const std::tuple<T...> &in, std::tuple<To...>) {
    return std::make_tuple(std::get<To>(in)...);
  }
};

}

template <template <typename...> class predicate, typename T>
struct filter_tuple_values;

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, std::tuple<T...>>
    : detail::filter_tuple_values_impl<predicate, T...> {};

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, const std::tuple<T...>>
    : detail::filter_tuple_values_impl<predicate, T...> {};

}  // namespace executor
