//
// Created by shrijit on 6/4/20.
//

#include <array>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>
#include <utility>


template<unsigned _s0>
constexpr unsigned prod() { return _s0; }

template<unsigned _s0, unsigned _s1, unsigned... _sizes>
constexpr unsigned prod()
{
  return prod<_s1, _sizes...>() * _s0;
};



template<unsigned _s0 = 0, unsigned... _sizes>
struct shape_t
{
  static constexpr unsigned dim_v = 1 + sizeof...(_sizes);

  constexpr unsigned dim() const { return dim_v; }

  static constexpr unsigned numel_v = prod<_s0, _sizes...> ();

  constexpr unsigned numel() const { return numel_v; }

  template<unsigned _pos>
  static constexpr unsigned get(){ return std::get<_pos> (sizes); }

  unsigned get(unsigned pos) const { return sizes[pos]; }

  //it might be worth to have this bellow for runtime ease
  static constexpr std::array<unsigned, sizeof...(_sizes) + 1> sizes {_s0, _sizes...};
};

template<>
struct shape_t<0>
{
  std::vector<unsigned> sizes;

  shape_t () = default;

  shape_t(const std::initializer_list<unsigned>& l) : sizes(l) {};

  unsigned dim() const { return sizes.size(); }

  unsigned numel() const { return std::accumulate(sizes.cbegin(), sizes.cend(), 1, std::multiplies<unsigned>()); }

  unsigned get(unsigned pos) const { return sizes[pos]; }
};

//
//
//int main()
//{
//  // Compile time
//  std::cout << "Compile time static:\n"
//            << "shape_t<2,3,4>::dim_v: " << shape_t<2,3,4>::dim_v << '\n'
//            << "shape_t<2,3,4>::numel_v: " << shape_t<2,3,4>::numel_v << '\n'
//            << "shape_t<2,3,4>::get<0>(): " << shape_t<2,3,4>::get<0>() << '\n'
//            << "shape_t<2,3,4>::get<2>(): " << shape_t<2,3,4>::get<2>() << '\n';
//
//  // As instance methods
//  shape_t<2,3,4> shape_static;
//
//  std::cout << "Compile time static instance methods: shape_t<2,3,4> shape_static\n"
//            << "shape_static.dim(): " << shape_static.dim() << '\n'
//            << "shape_static.numel(): " << shape_static.numel() << '\n'
//            << "shape_static.get(0): " << shape_static.get(0) << '\n'
//            << "shape_static.get(2): " << shape_static.get(2) << '\n';
//
//  shape_t<> shape {2, 3, 4};
//  std::cout << "Runtime mutable shape: shape_t<> shape {2, 3, 4}\n"
//            << "shape.dim(): " << shape.dim() << '\n'
//            << "shape.numel(): " << shape.numel() << '\n'
//            << "shape.get(0): " << shape.get(0) << '\n'
//            << "shape.get(2): " << shape.get(2) << '\n';
//  return 0;
//}