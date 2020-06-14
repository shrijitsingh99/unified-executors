//
// Created by Shrijit Singh on 2020-06-14.
//

#pragma once

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <array>
#include <memory>

#include <execution/property.hpp>
#include <execution/type_trait.hpp>


template <template <typename, typename, typename, typename> typename Derived,
    typename Interface, typename Cardinality, typename Blocking,
    typename ProtoAllocator>
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
};
