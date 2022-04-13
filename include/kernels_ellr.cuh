#ifndef _KERNELS_ELLR_H_
#define _KERNELS_ELLR_H_

#include "compute_utils.cuh"

/**
 * @brief ELLR SpMV, adapted from CSR-Vector kernel by N. Bell & M. Garland. See CUSP library.
 * https://github.com/cusplibrary/cusplibrary/blob/develop/cusp/system/cuda/detail/multiply/csr_vector_spmv.h
 *
 * @tparam TIN type of the matrix and the dense vector (left-hand side)
 * @tparam TOUT type of the target vector (right-hand side)
 * @param M number of rows
 * @param R max. number of nnz in a row
 * @param Ax value iterator 1, matrix values
 * @param x value iterator 2, dense vector
 * @param Aj column iterator
 * @param Arl row lengths
 * @param y value iterator 3, result vector
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */

template <typename TIN, typename TOUT, unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_CUSP(const unsigned int M, const unsigned int R, const TIN *Ax, TIN *x,
                                             const int *Aj, const int *Arl, TOUT *y, const bool isAccumulative = false);

/**
 * @brief Entry-wise mixed-precision ELLR SpMV, adapted from CSR Vector kernel by K. Ahmad et al. (2019)
 *
 * @param M  number of rows
 * @param RS max. number of nnz in a row
 * @param AxS float value iterator 1, matrix values
 * @param xS  float value iterator 2, dense vector
 * @param AjS floats column iterator
 * @param ArlS floats row lengths
 * @param RD max. number of nnz in a row
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param AjD doubles column iterator
 * @param ArlD doubles row lengths
 * @param y value iterator 3, result vector
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Entrywise_Split(const unsigned int M, const unsigned int RS,
                                                              const float *AxS, float *xS, const int *AjS,
                                                              const int *ArlS, const unsigned int RD, const double *AxD,
                                                              const double *xD, const int *AjD, const int *ArlD,
                                                              double *y, const bool isAccumulative = false);

/**
 * @brief Row-wise split mixed-precision ELLR Vector kernel.
 *
 * @param M  number of rows
 * @param RS max. number of nnz in a row
 * @param AxS float value iterator 1, matrix values
 * @param xS  float value iterator 2, dense vector
 * @param AjS floats column iterator
 * @param ArlS floats row lengths
 * @param RD max. number of nnz in a row
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param AjD doubles column iterator
 * @param ArlD doubles row lengths
 * @param y value iterator 3, result vector
 * @param sep the first row where the precision is FP64, i.e. separation index
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Split(const unsigned int M, const unsigned int RS,
                                                            const float *AxS, float *xS, const int *AjS,
                                                            const int *ArlS, const unsigned int RD, const double *AxD,
                                                            const double *xD, const int *AjD, const int *ArlD,
                                                            double *y, const int sep,
                                                            const bool isAccumulative = false);

/**
 * @brief Row-wise dual mixed-precision ELLR Vector kernel.
 *
 * @param M number of rows
 * @param R max. number of nnz in a row
 * @param AxS float value iterator 1, matrix values
 * @param xS float value iterator 2, dense vector
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param Aj doubles column iterator
 * @param Arl doubles row lengths
 * @param y value iterator 3, result vector
 * @param rUseSingle row splits information
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual(const unsigned int M, const unsigned int R, const float *AxS,
                                                           float *xS, const double *AxD, const double *xD,
                                                           const int *Aj, const int *Arl, double *y,
                                                           const bool *rUseSingle, const bool isAccumulative = false);

/**
 * @brief Row-wise composite clustered mixed-precision ELLR Vector kernel.
 *
 * @param M number of rows
 * @param R max. number of nnz in a row
 * @param AxS float value iterator 1, matrix values
 * @param xS float value iterator 2, dense vector
 * @param AxD double value iterator 1, matrix values
 * @param xD double value iterator 2, dense vector
 * @param Aj doubles column iterator
 * @param Arl doubles row lengths
 * @param y value iterator 3, result vector
 * @param sep the first row where the precision is FP64, i.e. separation index
 * @param isAccumulative if true then does y += Ax, otherwise y = Ax
 */
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite(const unsigned int M, const unsigned int R,
                                                                const float *AxS, float *xS, const double *AxD,
                                                                const double *xD, const int *Aj, const int *Arl,
                                                                double *y, const int sep,
                                                                const bool isAccumulative = false);

#include "kernels_ellr.tpp"

#endif  // _KERNELS_ELLR_H_