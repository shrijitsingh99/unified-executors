//
// Created by shrijit on 5/31/20.
//

#pragma once

#include <Eigen/Dense>
#include <array>
#include <execution/executor/cuda_executor.hpp>

using namespace Eigen;

void mmul_gpu(cuda_executor<oneway_t, bulk_t, blocking_t::always_t> &ex,
              MatrixXd &a, MatrixXd &b, MatrixXd &c);
