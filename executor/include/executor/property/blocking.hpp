/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <executor/property/base_property.hpp>

namespace executor {

struct blocking_t : basic_executor_property<blocking_t, false, false> {
  friend constexpr bool operator==(const blocking_t& a, const blocking_t& b) {
    return a.which_ == b.which_;
  }

  friend constexpr bool operator!=(const blocking_t& a, const blocking_t& b) {
    return !(a == b);
  }

  constexpr blocking_t() : which_{0} {};

  struct always_t : basic_executor_property<always_t, true, true> {
    template <class Executor>
    friend Executor require(Executor& ex, const always_t& t) {
      return ex.require(t);
    }
  };

  static constexpr always_t always{};
  constexpr blocking_t(const always_t&) : which_{1} {};

  struct never_t : basic_executor_property<never_t, true, true> {
    template <class Executor>
    friend Executor require(Executor& ex, const never_t& t) {
      return ex.require(t);
    }
  };

  static constexpr never_t never{};
  constexpr blocking_t(const never_t&) : which_{2} {};

  struct possibly_t : basic_executor_property<possibly_t, true, true> {
    template <class Executor>
    friend Executor require(Executor& ex, const possibly_t& t) {
      return ex.require(t);
    }
  };

  static constexpr possibly_t possibly{};
  constexpr blocking_t(const possibly_t&) : which_{3} {};

  int which_;
};

static constexpr blocking_t blocking{};

// Can inline member variable in C++17 eliminating need for sperate CPP file
// https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
// inline constexpr blocking_t::possibly_t blocking_t::possibly;

}  // namespace executor
