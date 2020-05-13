//
// Created by Shrijit Singh on 2020-05-11.
//

#include <iostream>
#include <vector>

struct false_type {
  static const bool value = false;
};

struct true_type {
  static const bool value = true;
};


template <typename T, typename U>
struct is_same: false_type {};

template <typename T>
struct is_same<T, T>: true_type {};


template <typename T>
struct is_floating_point: false_type {};

template <>
struct is_floating_point<float>: true_type{};

template <>
struct is_floating_point<double >: true_type{};

template<typename...>
using void_t = void;

template <typename T, typename = void>
struct is_incrementable: false_type{};

template <typename T>
struct is_incrementable<T, void_t<decltype(++std::declval<T&>())>>: true_type {};

template <typename T, typename = void>
struct is_member: false_type{};

int main(int argc, char *argv[]) {
  std::cout<<is_same<int, int>::value<<std::endl;
  std::cout<<is_same<int, double>::value<<std::endl;

  std::cout<<is_floating_point<int>::value<<std::endl;
  std::cout<<is_floating_point<double>::value<<std::endl;

  std::cout<<is_incrementable<double>::value<<std::endl;
  std::cout<<is_incrementable<std::vector<int>>::value<<std::endl;

}