//
// Created by shrijit on 6/20/20.
//

#pragma once

#include <executor/trait/common_traits.hpp>
#include <type_traits>

namespace executor {
template <typename Executor, typename Property,
          typename std::enable_if_t<
              Property::template is_applicable_property_v<Executor>, int> = 0>
constexpr auto query(Executor &&ex, const Property &p) noexcept {
  return Property::template static_query<remove_cv_ref_t<Executor>>();
}
}  // namespace executor
