/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <executor/executor.h>

using namespace executor;

int main() {
  auto exec1 = inline_executor<>{};
  auto exec2 = inline_executor<blocking_t::never_t>{};

  blocking_t::possibly_t::static_query<inline_executor<>>();
  std::cout << typeid(blocking_t::possibly_t::static_query<inline_executor<>>())
                   .name()
            << std::endl;

  std::cout << typeid(executor::blocking_t::always).name() << std::endl;
  if (exec1.query(executor::blocking) == executor::blocking.always)
    printf("Equal\n");

  if (exec1 == exec2) printf("Same\n");
}
