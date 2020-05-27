#pragma once

#include <functional>
#include <string>
#include <iostream>
#include <sfinae_traits.hpp>

template<bool requireable>
class basic_executor_property {
  static constexpr bool is_requirable = requireable;

  template<class T>
  static constexpr bool is_applicable_property() {
    return is_executor<T>::value;
  }

  template<typename T>
  static constexpr bool is_applicable_property_v = is_applicable_property<T>();
};


template<class Executor>
class executor_shape
{
 private:
  // exposition only
  template<class T>
  using helper = typename T::shape_type;

 public:
  using type = std::experimental::detected_or_t<size_t, helper, Executor>;

  // exposition only
  static_assert(std::is_integral_v<type>, "shape type must be an integral type");
};


template<class Executor> struct executor_shape;
template<class Executor> using executor_shape_t = typename executor_shape<Executor>::type;



class blocking_t {
 public:
  class always_t: basic_executor_property<true> {
   public:
    template<class Executor>
    friend Executor require(Executor &ex, const always_t &) {
      return ex;
    }
    template<class Executor>
    friend always_t query(const Executor &ex, const always_t &) {
      return ex.blocking();
    }
  };

  class never_t: basic_executor_property<true> {
   public:
    template<class Executor>
     friend Executor require(Executor &ex, const never_t &) {
      return ex;
    }
    template<class Executor>
    friend never_t query(const Executor &ex, const never_t &) {
      return ex.blocking();
    }
  };

};
