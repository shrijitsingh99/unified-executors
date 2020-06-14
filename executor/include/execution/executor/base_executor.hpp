//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <array>
#include <execution/property.hpp>
#include <execution/type_trait.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

template <template <typename...> typename Derived, typename Interface,
          typename Cardinality, typename Blocking, typename ProtoAllocator>
class executor {
 public:
  static constexpr bool query(const blocking_t::always_t &t) {
    return std::is_same<Blocking, blocking_t::always_t>();
  }

  static constexpr bool query(const blocking_t::never_t &t) {
    return std::is_same<Blocking, blocking_t::never_t>();
  }

  static constexpr bool query(const blocking_t::possibly_t &t) {
    return std::is_same<Blocking, blocking_t::possibly_t>();
  }

  static constexpr bool query(const oneway_t &t) {
    return std::is_same<Interface, oneway_t>();
  }

  static constexpr bool query(const twoway_t &t) {
    return std::is_same<Interface, twoway_t>();
  }

  template <typename F, typename... Args>
  void bulk_execute(F &&f, Args &&... args, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)..., i);
    }
  }
};
