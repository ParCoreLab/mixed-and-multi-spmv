#ifndef _KERNELS_CSR_H_
#define _KERNELS_CSR_H_

#include "compute_utils.cuh"

/**
 * @brief CSR Vector kernel by N. Bell & M. Garland. See CUSP library.
 * https://github.com/cusplibrary/cusplibrary/blob/develop/cusp/system/cuda/detail/multiply/csr_vector_spmv.h
 *
 * @tparam TIN type of the matrix and the dense vector (left-hand side)
 * @tparam TOUT type of the target vector (right-hand side)
 * @param M number of rows
 * @param Ax value iterator 1, matrix values
 * @param x value iterator 2, dense vector
 * @param Aj column iterator
 * @param Ap row iterator
 * @param y value iterator 3, result vector
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */

template <typename TIN, typename TOUT, unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_CSR_vector_CUSP(const unsigned int M, const TIN *Ax, TIN *x, const int *Aj, const int *Ap,
                                            TOUT *y, const bool isAccumulative = false);

/**
 * @brief Entry-wise mixed-precision CSR Vector kernel by K. Ahmad et al. (2019)
 *
 * @param M  number of rows
 * @param AxS float value iterator 1, matrix values
 * @param xS  float value iterator 2, dense vector
 * @param AjS floats column iterator
 * @param ApS floats row iterator
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param AjD doubles column iterator
 * @param ApD doubles row iterator
 * @param y value iterator 3, result vector
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 * @param isUpdatingXS if true, when the final result is also downcasted and written in xS
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_CSR_vector_Mixed_Entrywise_Split(const unsigned int M, const float *AxS, float *xS,
                                                             const int *AjS, const int *ApS, const double *AxD,
                                                             const double *xD, const int *AjD, const int *ApD,
                                                             double *y, const bool isAccumulative = false);

/**
 * @brief Row-wise split mixed-precision CSR Vector kernel.
 *
 * @param M  number of rows
 * @param AxS float value iterator 1, matrix values
 * @param xS  float value iterator 2, dense vector
 * @param AjS floats column iterator
 * @param ApS floats row iterator
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param AjD doubles column iterator
 * @param ApD doubles row iterator
 * @param y value iterator 3, result vector
 * @param sep the first row where the precision is FP64, i.e. separation index
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 * @param isUpdatingXS if true, when the final result is also downcasted and written in xS
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_CSR_vector_Mixed_Rowwise_Split(const unsigned int M, const float *AxS, float *xS,
                                                           const int *AjS, const int *ApS, const double *AxD,
                                                           const double *xD, const int *AjD, const int *ApD, double *y,
                                                           const int sep, const bool isAccumulative = false);

/**
 * @brief Row-wise dual mixed-precision CSR Vector kernel.
 *
 * @param M number of rows
 * @param AxS float value iterator 1, matrix values
 * @param xS float value iterator 2, dense vector
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param Aj doubles column iterator
 * @param Ap doubles row iterator
 * @param y value iterator 3, result vector
 * @param rUseSingle row splits information
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 * @param isUpdatingXS if true, when the final result is also downcasted and written in xS
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_CSR_vector_Mixed_Rowwise_Dual(const unsigned int M, const float *AxS, float *xS,
                                                          const double *AxD, const double *xD, const int *Aj,
                                                          const int *Ap, double *y, const bool *rUseSingle,
                                                          const bool isAccumulative = false);

/**
 * @brief Row-wise composite clustered mixed-precision CSR Vector kernel.
 *
 * @param M number of rows
 * @param AxS float value iterator 1, matrix values
 * @param xS float value iterator 2, dense vector
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param Aj doubles column iterator
 * @param Ap doubles row iterator
 * @param y value iterator 3, result vector
 * @param sep the first row where the precision is FP64, i.e. separation index
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 * @param isUpdatingXS if true, when the final result is also downcasted and written in xS
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_CSR_vector_Mixed_Rowwise_Composite(const unsigned int M, const float *AxS, float *xS,
                                                               const double *AxD, const double *xD, const int *Aj,
                                                               const int *Ap, double *y, const int sep,
                                                               const bool isAccumulative = false);

#include "kernels_csr.tpp"

#endif  // _KERNELS_CSR_H_