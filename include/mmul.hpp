//
// Created by Shrijit Singh on 2020-05-31.
//

#pragma once

#include "../../../../usr/local/include/eigen3/Eigen/Dense"

#if defined(_OPENMP)
#include "../../../../usr/lib/gcc/x86_64-linux-gnu/7/include/omp.h"
#endif


#include <execution/executor.hpp>
#include <mmul.cuh>

using namespace Eigen;

template <typename Interface, typename Blocking, typename ProtoAllocator>
void mmul(inline_executor<Interface, Blocking, ProtoAllocator> ex, MatrixXd &a,
          MatrixXd &b, MatrixXd &c) {
  c = a * b;
}

template <typename Interface, typename Blocking, typename ProtoAllocator>
void mmul(omp_executor<Interface, Blocking, ProtoAllocator> ex, MatrixXd &a,
          MatrixXd &b, MatrixXd &c) {

  auto mul = [=](MatrixXd &a, MatrixXd &b, MatrixXd &c,
                 std::size_t thread_idx) {
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


template <typename Interface, typename Blocking, typename ProtoAllocator>
void mmul(cuda_executor<Interface, Blocking, ProtoAllocator> ex, MatrixXd &a,
          MatrixXd &b, MatrixXd &c) {

  void *a_g, *b_g, *c_g;
  a_g = device_upload((void*) a.data(), a.size()*sizeof(double));
  b_g = device_upload((void*) b.data(), b.size()*sizeof(double));
  c_g = device_upload((void*) c.data(), c.size()*sizeof(double));

  std::vector<std::size_t> shape = {std::size_t(ceil(a.rows()/2.0)), std::size_t(ceil(b.cols()/2.0)), 1, 2, 2, 1};
  cuda_executor<oneway_t, blocking_t::always_t, void>{}.bulk_execute(
      mmul_gpu, shape, a_g, b_g, c_g, a.rows(), a.cols(), b.cols());

  memcpy((double*)c.data(), c_g, 9*sizeof(double));
}