#include <array>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

template <unsigned _s0> constexpr unsigned prod() { return _s0; }

template <unsigned _s0, unsigned _s1, unsigned... _sizes>
constexpr unsigned prod() {
  return prod<_s1, _sizes...>() * _s0;
};

template <unsigned _s0 = 0, unsigned... _sizes> struct shape_t {
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

//template <> struct shape_t<0> {
//  std::vector<unsigned> sizes;
//
//  shape_t() = default;
//
//  shape_t(const std::initializer_list<unsigned> &l) : sizes(l){};
//
//  unsigned dim() const { return sizes.size(); }
//
//  unsigned numel() const {
//    return std::accumulate(sizes.cbegin(), sizes.cend(), 1,
//                           std::multiplies<unsigned>());
//  }
//
//  unsigned get(unsigned pos) const { return sizes[pos]; }
//};

template <typename Shape, unsigned _size = Shape::dim()>
struct shape_wrapper_t: Shape {
  explicit shape_wrapper_t(Shape shape) {
    static_assert(_size == Shape::dim(), "Dim not matching");
  }
};