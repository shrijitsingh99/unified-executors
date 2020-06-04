#include <array>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>


template <typename Shape, unsigned _size = Shape::dim()>
struct shape_wrapper_t {
  shape_wrapper_t() = default;

  unsigned get(unsigned pos) const { return Shape::sizes[pos]; }

  unsigned operator[](unsigned dim) { return get(dim); }
};

template <unsigned _s0> constexpr unsigned prod() { return _s0; }

template <unsigned _s0, unsigned _s1, unsigned... _sizes>
constexpr unsigned prod() {
  return prod<_s1, _sizes...>() * _s0;
};

template <unsigned _s0 = 0, unsigned... _sizes> struct shape_t: shape_wrapper_t<shape_t<_s0, _sizes...>, 1 + sizeof...(_sizes)> {

  shape_t() = default;

  static constexpr unsigned dim_v = 1 + sizeof...(_sizes);

  constexpr static unsigned dim() { return dim_v; }

  static constexpr unsigned numel_v = prod<_s0, _sizes...>();

  constexpr unsigned numel() const { return numel_v; }

  template <unsigned _pos> static constexpr unsigned get() {
    return std::get<_pos>(sizes);
  }

  unsigned get(unsigned pos) const { return sizes[pos]; }

  unsigned operator[](unsigned dim) { return get(dim); }

  // it might be worth to have this bellow for runtime ease
  static constexpr std::array<unsigned, sizeof...(_sizes) + 1> sizes{_s0,
                                                                     _sizes...};
};

