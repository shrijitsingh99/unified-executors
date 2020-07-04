//
// Created by Shrijit Singh on 18/06/20.
//

#pragma once

#include <type_traits>

namespace execution {

// Part of Standard Library in C++17 onwards

template <typename...>
using void_t = void;

template <typename T>
using remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;

}  // namespace execution
