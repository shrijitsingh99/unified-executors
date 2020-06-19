#include <mmul.cuh>

__host__ void *device_upload(void *var, std::size_t size) {
  void *gpu_var;
  cudaMallocManaged(&gpu_var, size);
  memcpy(gpu_var, var, size);
  return gpu_var;
}

__device__ void mmul_gpu(double *a, double *b, double *c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0;
  if (col < k && row < m) {
    for (int i = 0; i < n; i++) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

void mmul_gpu_host(MatrixXd &a, MatrixXd &b, MatrixXd &c) {
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

  int rows1 = a.rows(), cols1 = a.cols(), cols2 = b.cols();

  cuda_executor<oneway_t, bulk_t, blocking_t::always_t, void>{}.bulk_execute(
      [=] __device__(void) { mmul_gpu(a_g, b_g, c_g, rows1, cols1, cols2); },
      shape);

  memcpy(static_cast<double *>(c.data()), c_g, 9 * sizeof(double));
}
