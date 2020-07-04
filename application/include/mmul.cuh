/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <Eigen/Dense>
#include <array>
#include <executor/default/cuda_executor.hpp>

using namespace Eigen;

void mmul_gpu(const executor::cuda_executor<>& ex, const MatrixXd& a,
              const MatrixXd& b, MatrixXd& c);
