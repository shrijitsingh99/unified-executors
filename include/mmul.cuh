//
// Created by shrijit on 5/31/20.
//

#pragma once

__global__ void mmul_gpu(double *a, double* b, double* c, int m, int n, int k);
