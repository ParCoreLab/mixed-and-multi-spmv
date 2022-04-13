#ifndef _COMPUTE_UTILS_H_
#define _COMPUTE_UTILS_H_

#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include <cfloat>

// Kernel specific
#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS_MAX 65535

#include "kernels.cuh"
#include "kernels_csr.cuh"
#include "kernels_ellr.cuh"
#include "matrix.hpp"
#include "parameters.hpp"
#include "precisions.hpp"
#include "printing.hpp"
#include "verify.hpp"

// SpMV Macros
#define SPMV_INITIAL_Y_VALUE 0.0
#define SPMV_INITIAL_X_RANGE_LOWER -5.0
#define SPMV_INITIAL_X_RANGE_UPPER 5.0
#define SPMV_IS_ACCUMULATIVE true  // (T) : y <- y + Ax | (F) : y <- Ax
// Jacobi Macros
#define JACOBI_SKIP_THRESHOLD_ERROR 1e-1
#define JACOBI_INITIAL_X_VALUE 0.0
// Cardiac Macros
#define CARDIAC_INITIAL_X_VALUE 1.1

// if delta change is in range (L, U) we break for the next level of precision
#define JACOBI_BREAKPOINT_EPSILON_L 0.9
#define JACOBI_BREAKPOINT_EPSILON_U 1.1

#define SETSTEPS(steps, fp32, mixed, fp64) \
  {                                        \
    steps[0] = (fp32);                     \
    steps[1] = (mixed);                    \
    steps[2] = (fp64);                     \
  }
// Taken from https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_gpu/jacobi.cu
#define CUDA_CHECK_CALL(call)                                                                                     \
  {                                                                                                               \
    cudaError_t cudaStatus = call;                                                                                \
    if (cudaSuccess != cudaStatus)                                                                                \
      fprintf(stderr, "error CUDA RT call \"%s\" in line %d of file %s failed with  %s (%d).\n", #call, __LINE__, \
              __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);                                              \
  }

// Check kernel errors if it was not specified by makefile
#ifndef CUDA_CHECK_KERNELS
#define CUDA_CHECK_KERNELS 1
#endif
// Check the success of a kernel. Since they dont return an error code, we check with cudaGetLastError instead.
#if CUDA_CHECK_KERNELS
#define CUDA_GET_LAST_ERR(label)                                          \
  {                                                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(err)); \
    }                                                                     \
  }
#else
#define CUDA_GET_LAST_ERR(label) ;
#endif

// Start CUDA events for the stopwatch
// double start = omp_get_wtime(), stop;
#define START_TIMERS()                        \
  cudaEvent_t start, stop;                    \
  CUDA_CHECK_CALL(cudaEventCreate(&(start))); \
  CUDA_CHECK_CALL(cudaEventCreate(&(stop)));  \
  CUDA_CHECK_CALL(cudaEventRecord(start, 0));

// Stop CUDA events for the stopwatch, and record the milliseconds passed.
// CUDA_CHECK_CALL(cudaDeviceSynchronize());
// stop = omp_get_wtime();
// ms = (stop - start) / 1000.0;
#define STOP_TIMERS(ms)                                          \
  CUDA_CHECK_CALL(cudaEventRecord(stop, 0));                     \
  CUDA_CHECK_CALL(cudaEventSynchronize(stop));                   \
  CUDA_CHECK_CALL(cudaEventElapsedTime(&(ms), (start), (stop))); \
  CUDA_CHECK_CALL(cudaEventDestroy(start));                      \
  CUDA_CHECK_CALL(cudaEventDestroy(stop));

// permutation type
typedef enum {
  PERMUTE_FORWARD,   // row i goes to perm[i]
  PERMUTE_BACKWARD,  // row perm[i] goes to i
} perm_type_vector_t;
typedef enum {
  PERMUTE_ROWS,      // row i goes to perm[i]
  PERMUTE_SYMMETRIC  // both row and col i goes to perm[i]
} perm_type_matrix_t;

/**
 * @brief Return the blocks per grid for SpMV vector CSR kernel.
 *
 * @param rows number of rows
 * @param avg_nz_inrow average number of non-zeros in a row
 * @return blocks per grid
 */
size_t spmv_grid_size(int rows, double avg_nz_inrow);

/**
 * @brief Return the blocks per grid for kernels over a vector.
 *
 * @param len length of the vector
 * @return blocks per grid
 */
size_t vector_kernels_grid_size(int len);

/**
 * @brief Creates an spmv evaluation object
 *
 * @param rows number of rows
 * @param nzD number of double non-zero values
 * @param nzS number of single non-zero values
 * @param iter number of iterations
 * @param milliseconds average SpMV runtie in milliseconds
 * @return evaluation object
 */
eval_t create_spmv_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds);

/**
 * @brief Creates a cardiac simulation evaluation object
 *
 * @param rows number of rows
 * @param nzD number of double non-zero values
 * @param nzS number of single non-zero values
 * @param iter number of iterations
 * @param milliseconds average SpMV runtie in milliseconds
 * @return evaluation object
 */
eval_t create_cardiac_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds);

/**
 * @brief Create a jacobi evaluation object
 *
 * @param rows number of rows
 * @param nzD number of double non-zero values
 * @param nzS number of single non-zero values
 * @param iter number of iterations
 * @param milliseconds average SpMV runtie in milliseconds
 * @param delta final delta value
 * @param error final error value (residual)
 * @param isConverged did iterative solver converge?
 * @return evaluation object
 */
eval_t create_jacobi_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds, double delta, double error,
                                bool isConverged);

/**
 * @brief Prepares a vector to be arr[i] = i / |a| (1-indexed)
 *
 * @param arr vector
 * @param size length
 */
void jacobi_reset_vector(double *arr, const int size);

/**
 * @brief Given a matrix, entry-wise split it into 2 separate matrices.
 */
void split_entrywise(CSR_Matrix<double> *mat, precision_e *p, CSR_Matrix<float> *matS, CSR_Matrix<double> *matD);
void split_entrywise(ELLR_Matrix<double> *mat, precision_e *p, ELLR_Matrix<float> *matS, ELLR_Matrix<double> *matD);

/**
 * @brief Given a matrix, row-wise split it into 2 separate matrices.
 */
void split_rowwise(CSR_Matrix<double> *mat, precision_e *p, CSR_Matrix<float> *matS, CSR_Matrix<double> *matD,
                   perm_type_matrix_t perm_type);
void split_rowwise(ELLR_Matrix<double> *mat, precision_e *p, ELLR_Matrix<float> *matS, ELLR_Matrix<double> *matD,
                   perm_type_matrix_t perm_type);

/**
 * @brief From a rowwise clustering, find how many nonzero values are stored in which precision.
 * Also returns the number of empty rows.
 */
void get_nonzero_counts_rowwise(CSR_Matrix<double> *mat, precision_e *p, int *nzD, int *nzS, int *mE);
void get_nonzero_counts_rowwise(ELLR_Matrix<double> *mat, precision_e *p, int *nzD, int *nzS, int *mE);

/**
 * @brief Given a rowwise precision array, create the permutation mapping array. Given output array is assumed to be
 * allocated.
 *
 * @param p row precisions
 * @param n number of rows
 * @param ans permutation array
 */
void get_permutations(const precision_e *p, const int n, int *ans);

/**
 * @brief Checks the convergence details of FP64 Jacobi solver, skips the rest if it is bad.
 * Note that it does not check if it is converged, it only checks if the delta and error is usable at the end.
 * Also checks if it took more than a few iterations to converge in case it does.
 *
 * @param evals Jacobi evaluation
 * @return true if it is bad, false otherwise.
 */
bool check_bad_convergence(eval_t evals);

template <typename T>
void permute_matrix(CSR_Matrix<T> *mat, const precision_e *p, CSR_Matrix<T> *matnew, int *sep,
                    perm_type_matrix_t perm_type);
template <typename T>
void permute_matrix(ELLR_Matrix<T> *mat, const precision_e *p, ELLR_Matrix<T> *matnew, int *sep,
                    perm_type_matrix_t perm_type);

template <typename T>
void permute_vector(const T *vec, const int len, const precision_e *p, T *vecnew, perm_type_vector_t perm_type);

template <typename T>
T sum_vector(T *vec, int N);

/**
 * @brief SpMV of a sparse matrix in CSR format with it's diagonal provided separately. Only one iteration.
 *
 * @param mat CSR matrix without the diagonal
 * @param diag diagonal as an array
 * @param x dense vector
 * @param y result
 * @param perm if permuting the matrix is required, give it here. Defaults to NULL
 */
void spmv_CSR_FP64_with_Diagonal(CSR_Matrix<double> *mat, const double *diag, double *x, double *y);

/**
 * @brief Finds the checkpoints for a multi-precision algorithm.
 * steps  [x  x  x]
 *         |  |  |-> do FP64?
 *         |  |----> do MIXED?
 *         |-------> do FP32?
 *
 * @param iters total number of iterations to divide
 * @param steps which steps should it do?
 * @param checkpoints checpoints will work as follows:
 *    if (i < checkpoints[0]):
 *      ... (FP32)
 *    elif (i < checkpoints[1]):
 *      ... (MIXED)
 *    else:
 *      ... (FP64)
 *
 */
void find_checkpoints(const int iters, const bool steps[3], int checkpoints[2]);

/**
 * @brief Calculate potential occupancy of a kernel
 *
 * @param kernel pointer to the kernel function
 * @return ratio of active warps to maximum warps
 */
double report_potential_occupancy(void *kernel);

/**
 * @brief Print potential occupancies of kernels
 * using report_potential_occupancy
 */
void show_kernel_occupancies();

#include "compute_utils.tpp"

#endif  // _COMPUTE_UTILS_H_