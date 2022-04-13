#include "kernels.cuh"

__global__ void kernel_jacobi_iteration(const double *y, const double *diag, double *xbar, const int M) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < M) {
    xbar[i] = (y[i] - xbar[i]) / diag[i];  // store new guess
    i += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_cardiac_update(const double *y, const double *diag, double *xD, float *xS, const int M) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < M) {
    const double tmp = y[i] + xD[i] * diag[i];
    xD[i] = tmp;
    if (xS) xS[i] = (float)tmp;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_jacobi_iterate_and_update(const double *y, const double *diag, const double *xbar, double *xD,
                                                 float *xS, const int M) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < M) {
    const double tmp_xbar = (y[i] - xbar[i]) / diag[i];  // find new guess
    xD[i] = tmp_xbar;                                    // update fp64 x always
    if (xS) xS[i] = tmp_xbar;                            // update fp32 x (if required)
    i += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_add_diag(const double *x, const double *diag, double *y, const int M) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < M) {
    y[i] += x[i] * diag[i];
    i += blockDim.x * gridDim.x;
  }
}