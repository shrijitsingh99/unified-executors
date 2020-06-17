//
// Created by Shrijit Singh on 14/06/20.
//

#pragma once

namespace execution {

template <template <typename, typename, typename, typename> typename T>
struct is_executor_available : std::false_type {};

template <template <typename, typename, typename, typename> typename T>
using is_executor_available_t = typename is_executor_available<T>::type;

}  // namespace execution
