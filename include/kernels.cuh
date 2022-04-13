#ifndef _KERNELS_H_
#define _KERNELS_H_

/**
 * @brief Do the cardiac update (swapping)
 *
 * @param y right-hand side vector
 * @param diag diagonal
 * @param xD double dense vector
 * @param xS single dense vector
 * @param M number of rows
 */
__global__ void kernel_cardiac_update(const double *y, const double *diag, double *xD, float *xS, const int M);

/**
 * @brief Do the jacobi iteration (y - Ex) / d, where Ex is stored in xbar.
 *
 * @param y right hand-side vector
 * @param diag diagonal
 * @param xbar temporary vector storing E*x
 * @param M number of rows
 */
__global__ void kernel_jacobi_iteration(const double *y, const double *diag, double *xbar, const int M);

/**
 * @brief Updates the old solutions via Jacobi Iteration.
 *
 * @param y right hand-side vector
 * @param diag diagonals
 * @param xbar result of SpMV
 * @param xD FP64 solutions to be updated
 * @param xS FP32 solutions to be updated
 * @param M number of rows
 */
__global__ void kernel_jacobi_iterate_and_update(const double *y, const double *diag, const double *xbar, double *xD,
                                                 float *xS, const int M);

/**
 * @brief Suppose Ex is calcualted at y, then finds Ax by y[i] += d[i] * x[i].
 *
 * @param x dense vector
 * @param diag diagonal
 * @param y output vector
 * @param M number of rows
 */
__global__ void kernel_add_diag(const double *x, const double *diag, double *y, const int M);

#endif  // _KERNELS_H_