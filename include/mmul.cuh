//
// Created by shrijit on 5/31/20.
//

#pragma once

#ifdef CUDA
__global__
#endif
void mmul_gpu(double *a, double* b, double* c, int m, int n, int k);
