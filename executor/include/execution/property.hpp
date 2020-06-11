#pragma once

#include "type_traits.hpp"
#include <functional>
#include <iostream>
#include <string>

// Base Property
template <typename Derived, bool requireable, bool preferable>
class basic_executor_property {
public:
  static constexpr bool is_requirable = requireable;
  static constexpr bool is_preferable = preferable;

  template <class T> static constexpr bool is_applicable_property() {
    return execution::is_executor<T>();
  }

  template <class Executor>
  static constexpr auto static_query()
      -> decltype(Executor::query(std::declval<Derived>())) {
    return Executor::query(Derived{});
  }

  template <typename T>
  static constexpr bool is_applicable_property_v = is_applicable_property<T>();

  template <class Executor>
  static constexpr decltype(auto) static_query_v = static_query<Executor>();
};

// Shape Property
template <class Executor>
class executor_shape: basic_executor_property<executor_shape<Executor>, true, true> {
//private:
//  template <class T> using helper = typename T::shape_type;

public:
  template<unsigned _s0 = 0, unsigned... _sizes>
  using type = std::size_t;
//  using type = std::experimental::detected_or_t<size_t, helper, Executor>;
//
//  static_assert(std::is_integral_v<type>,
//                "shape type must be an integral type");
};

template <class Executor> struct executor_shape;

template <class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;

// Blocking Property
class blocking_t {
public:
  class always_t : basic_executor_property<always_t, true, true> {
  public:
    template <class Executor>
    friend Executor require(Executor &ex, const always_t &t) {
      return ex.require(t);
    }
    template <class Executor>
    friend bool query(const Executor &ex, const always_t &t) {
      return std::is_same<always_t, decltype(ex.interface)>();
    }
  };

  static constexpr always_t always{};

  class never_t : basic_executor_property<never_t, true, true> {
  public:
    template <class Executor>
    friend Executor require(Executor &ex, const never_t &t) {
      return ex.require(t);
    }
    template <class Executor>
    friend bool query(const Executor &ex, const never_t &t) {
      return std::is_same<never_t, decltype(ex.interface)>();
    }
  };

  static constexpr never_t never{};

  class possibly_t : basic_executor_property<possibly_t, true, true> {
  public:
    template <class Executor>
    friend Executor require(Executor &ex, const possibly_t &t) {
      return ex.require(t);
    }
    template <class Executor>
    friend bool query(const Executor &ex, const possibly_t &t) {
      return std::is_same<never_t, decltype(ex.interface)>();
    }
  };

  static constexpr possibly_t possibly{};
};

static constexpr blocking_t blocking;

// Allocator Property
template <typename ProtoAllocator>
class allocator_t
    : basic_executor_property<allocator_t<ProtoAllocator>, true, true> {
public:
  constexpr explicit allocator_t(const ProtoAllocator &alloc) : alloc_(alloc) {}

  constexpr ProtoAllocator value() const { return alloc_; }

private:
  ProtoAllocator alloc_;
};

template <>
struct allocator_t<void>
    : basic_executor_property<allocator_t<void>, true, true> {
  template <class ProtoAllocator>
  constexpr allocator_t<ProtoAllocator>
  operator()(const ProtoAllocator &alloc) const {
    return allocator_t<ProtoAllocator>{alloc};
  }
};

static constexpr allocator_t<void> allocator;

// Direction Property
class oneway_t : basic_executor_property<oneway_t, true, true> {
  template <typename Executor>
  friend Executor require(Executor &ex, const oneway_t &t) {
    return ex.require(t);
  }
  template <class Executor>
  friend bool query(const Executor &ex, const oneway_t &t) {
    return std::is_same<oneway_t, decltype(ex.interface)>();
  }
};

static constexpr oneway_t oneway;

class twoway_t : basic_executor_property<twoway_t, true, true> {
  template <typename Executor>
  friend Executor require(Executor &ex, const twoway_t &t) {
    return ex.require(t);
  }
  template <class Executor>
  friend bool query(const Executor &ex, const twoway_t &t) {
    return std::is_same<twoway_t, decltype(ex.interface)>();
  }
};

static constexpr twoway_t twoway;

// Bulk Property
class single_t : basic_executor_property<single_t, true, true> {
  template <typename Executor>
  friend Executor require(Executor &ex, const single_t &t) {
    return ex.require(t);
  }
  template <class Executor>
  friend bool query(const Executor &ex, const single_t &t) {
    return std::is_same<single_t, decltype(ex.interface)>();
  }
};

static constexpr single_t single;

class bulk_t : basic_executor_property<bulk_t, true, true> {
  template <typename Executor>
  friend Executor require(Executor &ex, const bulk_t &t) {
    return ex.require(t);
  }
  template <class Executor>
  friend bool query(const Executor &ex, const bulk_t &t) {
    return std::is_same<bulk_t, decltype(ex.interface)>();
  }
};

static constexpr bulk_t bulk;
