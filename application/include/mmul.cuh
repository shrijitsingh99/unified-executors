//
// Created by shrijit on 5/31/20.
//

#pragma once

#include <Eigen/Dense>
#include <array>
#include <execution/executor/cuda_executor.hpp>

using namespace Eigen;

void mmul_gpu(const cuda_executor<> &ex, const MatrixXd &a, const MatrixXd &b,
              MatrixXd &c);
