//
// Created by Shrijit Singh on 2020-05-31.
//

#pragma once

#include <Eigen/Dense>
#include <array>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <execution/executor.hpp>
#include <mmul.cuh>

using namespace Eigen;

template <typename Executor,
          typename execution::instance_of_base<inline_executor, Executor> = 0>
void mmul(Executor ex, MatrixXd &a, MatrixXd &b, MatrixXd &c) {
  auto mul = [&]() { c = a * b; };

  ex.execute(mul);
}

template <typename Executor,
          typename execution::instance_of_base<omp_executor, Executor> = 0>
void mmul(Executor ex, MatrixXd &a, MatrixXd &b, MatrixXd &c) {
  auto mul = [&](std::size_t thread_idx) {
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
  ex.bulk_execute(mul, a.rows());
}

template <typename Executor,
          typename execution::instance_of_base<cuda_executor, Executor> = 0>
void mmul(Executor ex, MatrixXd &a, MatrixXd &b, MatrixXd &c) {
  mmul_gpu(ex, a, b, c);
}
