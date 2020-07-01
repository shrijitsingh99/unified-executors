//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <array>
#include <execution/property.hpp>
#include <execution/trait/is_executor_available.hpp>
#include <execution/trait/is_instance_of_base.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

template <template <typename...> class Derived, typename Interface,
          typename Cardinality, typename Blocking, typename ProtoAllocator>
class executor {
 public:
  template <typename Executor,
            typename execution::instance_of_base<Derived, Executor> = 0>
  bool operator==(const Executor &rhs) const noexcept {
    return std::is_same<
        Derived<Interface, Cardinality, Blocking, ProtoAllocator>,
        Executor>::value;
  }

  template <typename Executor,
            typename execution::instance_of_base<Derived, Executor> = 0>
  bool operator!=(const Executor &rhs) const noexcept {
    return !operator==(rhs);
        Derived<Interface, Cardinality, Blocking, ProtoAllocator>,
        Executor>::value;
  }

  static constexpr bool query(const blocking_t::always_t &t) noexcept {
    return std::is_same<Blocking, blocking_t::always_t>();
  }

  static constexpr bool query(const blocking_t::never_t &t) noexcept {
    return std::is_same<Blocking, blocking_t::never_t>();
  }

  static constexpr bool query(const blocking_t::possibly_t &t) noexcept {
    return std::is_same<Blocking, blocking_t::possibly_t>();
  }

  static constexpr bool query(const oneway_t &t) noexcept {
    return std::is_same<Interface, oneway_t>();
  }

  static constexpr bool query(const twoway_t &t) noexcept {
    return std::is_same<Interface, twoway_t>();
  }

  template <typename F, typename... Args>
  void bulk_execute(F &&f, Args &&... args, std::size_t n) const {
    for (std::size_t i = 0; i < n; ++i) {
      std::forward<F>(f)(std::forward<Args>(args)..., i);
    }
  }
};
