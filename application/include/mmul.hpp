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
  double *a_g, *b_g, *c_g;
  a_g = static_cast<double *>(
      device_upload(static_cast<void *>(a.data()), a.size() * sizeof(double)));
  b_g = static_cast<double *>(
      device_upload(static_cast<void *>(b.data()), b.size() * sizeof(double)));
  c_g = static_cast<double *>(
      device_upload(static_cast<void *>(c.data()), c.size() * sizeof(double)));

  std::array<int, 6> shape{static_cast<int>(ceil(a.rows() / 2.0)),
                           static_cast<int>(ceil(b.cols() / 2.0)),
                           1,
                           2,
                           2,
                           1};

  cuda_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}.bulk_execute(
      [=](int stream_id) {
        mmul_gpu(a_g, b_g, c_g, a.rows(), a.cols(), b.cols());
      },
      shape);

  memcpy(static_cast<double *>(c.data()), c_g, 9 * sizeof(double));
}
