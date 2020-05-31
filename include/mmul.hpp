//
// Created by Shrijit Singh on 2020-05-31.
//

#pragma once

#include <Eigen/Dense>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <execution/executor.hpp>

using namespace Eigen;

template <typename Interface, typename Blocking, typename ProtoAllocator>
void mmul(inline_executor<Interface, Blocking, ProtoAllocator> ex, MatrixXd &a,
          MatrixXd &b, MatrixXd &c) {
  c = a * b;
}

template <typename Interface, typename Blocking, typename ProtoAllocator>
void mmul(omp_executor<Interface, Blocking, ProtoAllocator> ex, MatrixXd &a,
          MatrixXd &b, MatrixXd &c) {

  auto mul = [=](MatrixXd &a, MatrixXd &b, MatrixXd &c, std::size_t thread_idx) {
  #pragma omp for schedule(static)
    for (int i = 0; i < a.rows(); i = i + 1) {
      for (int j = 0; j < b.cols(); j = j + 1) {
        c(i, j) = 0.0;
        for (int k = 0; k < a.cols(); k = k + 1) {
          c(i, j) += a(i, k) * b(k, j);
        }
      }
    }
  };
  ex.bulk_execute(mul, 1, a, b, c);
}


