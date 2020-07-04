/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <Eigen/Dense>
#include <mmul.hpp>

using namespace Eigen;

TEST_CASE("Matrix Multiplication") {
  double dataA[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9},
         dataB[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, dataC[9] = {0};
  MatrixXd a = Map<Matrix<double, 3, 3, RowMajor>>(dataA);
  MatrixXd b = Map<Matrix<double, 3, 3, RowMajor>>(dataB);
  MatrixXd c = Map<Matrix<double, 3, 3, RowMajor>>(dataC);

  MatrixXd ans(3, 3);
  ans << 30, 36, 42, 66, 81, 96, 102, 126, 150;

  SUBCASE("Inline Matrix Multiplication") {
    c.setZero();
    mmul(inline_executor<>{}, a, b, c);
    CHECK(c.isApprox(ans));
  }

#if _OPENMP
  SUBCASE("OMP Matrix Multiplication") {
    c.setZero();
    mmul(omp_executor<>{}, a, b, c);
    CHECK(c.isApprox(ans));
  }
#endif

#if CUDA
  SUBCASE("CUDA Matrix Multiplication") {
    c.setZero();
    mmul(cuda_executor<>{}, a, b, c);
    CHECK(c.isApprox(ans));
  }
#endif
}
