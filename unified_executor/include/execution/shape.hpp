//
// Created by shrijit on 6/4/20.
//

#include <array>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

template <unsigned _size = 1, unsigned... _sizes> struct shape_t {
  static_assert(_size >= sizeof...(_sizes), "Dim don't match");
  static constexpr unsigned num_dim = _size;

  constexpr unsigned dim() const { return num_dim; }

  unsigned operator[](unsigned dim) { return get(dim); }

  // static constexpr unsigned numel_v = prod<_sizes...> ();

  // constexpr unsigned numel() const { return numel_v; }

  template <unsigned _pos> static constexpr unsigned get() {
    return std::get<_pos>(sizes);
  }

  unsigned get(unsigned pos) const { return sizes[pos]; }

  // it might be worth to have this bellow for runtime ease
  static constexpr std::array<unsigned, num_dim> sizes{_sizes...};
};