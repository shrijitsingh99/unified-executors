#pragma once

#include <experimental/type_traits>
#include <type_traits>

struct executor {
};

template<typename T> inline constexpr bool is_executor_v = std::is_base_of_v<executor, T>;
template<typename T> using is_executor = std::is_base_of<executor, T>;