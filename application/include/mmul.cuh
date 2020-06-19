//
// Created by shrijit on 5/31/20.
//

#pragma once

#include <Eigen/Dense>
#include <array>
#include <execution/executor/cuda_executor.hpp>
#include <stdexcept>

using namespace Eigen;

#ifdef CUDA
__host__ void *device_upload(void *var, std::size_t size);
#else
void *device_upload(void *var, std::size_t size) {
  throw std::runtime_error("No GPU");
}
#endif

void mmul_gpu_host(MatrixXd &a, MatrixXd &b, MatrixXd &c);
