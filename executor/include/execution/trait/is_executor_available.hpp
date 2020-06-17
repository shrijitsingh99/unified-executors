//
// Created by Shrijit Singh on 14/06/20.
//

#pragma once

namespace execution {

template <template <typename...> typename Executor>
struct is_executor_available : std::false_type {};

template <template <typename...> typename Executor>
constexpr bool is_executor_available_v = is_executor_available<Executor>::value;

}  // namespace execution
