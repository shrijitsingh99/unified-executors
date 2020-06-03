//
// Created by shrijit on 6/4/20.
//

#include <array>

template <unsigned _size = 1, unsigned... _sizes> struct shape_t {
  static_assert(_size >= sizeof...(_sizes), "Dim don't match");
  static constexpr unsigned num_dim = _size;

  constexpr unsigned dim() const { return num_dim; }

  unsigned operator[](unsigned dim) { return get(dim); }

  template <unsigned _pos> static constexpr unsigned get() {
    return std::get<_pos>(sizes);
  }

  unsigned get(unsigned pos) const { return sizes[pos]; }

  // it might be worth to have this bellow for runtime ease
  static constexpr std::array<unsigned, num_dim> sizes{_sizes...};
};

template <unsigned _size> struct shape_t<_size> {
  std::array<unsigned, _size> sizes;

  shape_t() = default;

  unsigned dim() const { return sizes.size(); }

  unsigned get(unsigned pos) const { return sizes[pos]; }

  unsigned operator[](unsigned dim) { return get(dim); }
};