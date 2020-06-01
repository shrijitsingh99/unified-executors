#pragma once

#include "type_traits.hpp"
#include <functional>
#include <iostream>
#include <string>


// Base Property
template <bool requireable> class basic_executor_property {
  static constexpr bool is_requirable = requireable;

  template <class T> static constexpr bool is_applicable_property() {
    return execution::is_executor<T>();
  }

  template <typename T>
  static constexpr bool is_applicable_property_v = is_applicable_property<T>();
};


// Shape Property
template <class Executor> class executor_shape {
 private:
  template <class T> using helper = typename T::shape_type;

 public:
  using type = std::experimental::detected_or_t<size_t, helper, Executor>;

  static_assert(std::is_integral_v<type>,
                "shape type must be an integral type");
};

template <class Executor> struct executor_shape;

template <class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


// Blocking Property
class blocking_t {
 public:
  class always_t : basic_executor_property<true> {
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

  class never_t : basic_executor_property<true> {
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
};

static constexpr blocking_t blocking;


// Allocator Property
template <typename ProtoAllocator>
class allocator_t: basic_executor_property<true> {
 public:
  constexpr explicit allocator_t(const ProtoAllocator& alloc) : alloc_(alloc) {}

  constexpr ProtoAllocator value() const
  {
    return alloc_;
  }

 private:
  ProtoAllocator alloc_;
};

template<>
struct allocator_t<void>: basic_executor_property<true>
{
  template<class ProtoAllocator>
  constexpr allocator_t<ProtoAllocator> operator()(const ProtoAllocator& alloc) const
  {
    return allocator_t<ProtoAllocator>{alloc};
  }
};

static constexpr allocator_t<void> allocator;


// Direction Property
class oneway_t: basic_executor_property<true> {
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

class twoway_t: basic_executor_property<true> {
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


// Bulk Direction Property
class bulk_oneway_t: basic_executor_property<true> {
  template <typename Executor>
  friend Executor require(Executor &ex, const bulk_oneway_t &t) {
    return ex.require(t);
  }
  template <class Executor>
  friend bool query(const Executor &ex, const bulk_oneway_t &t) {
    return std::is_same<bulk_oneway_t, decltype(ex.interface)>();
  }
};

static constexpr bulk_oneway_t bulk_oneway;


