//
// Created by shrijit on 6/9/20.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <iostream>
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
  ans << 30,  36,  42,
         66,  81,  96,
         102, 126, 150;

  SUBCASE("Inline Matrix Multiplication") {
    c.setZero();
    mmul(inline_executor<oneway_t, single_t, blocking_t::always_t, void>{}.decay_t(), a, b, c);
    CHECK(c.isApprox(ans));
  }

  SUBCASE("OMP Matrix Multiplication") {
    c.setZero();
    mmul(omp_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}, a, b, c);
    CHECK(c.isApprox(ans));
  }
  SUBCASE("CUDA Matrix Multiplication") {
    c.setZero();
    mmul(cuda_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}.decay_t(), a, b, c);
    CHECK(c.isApprox(ans));
  }

}