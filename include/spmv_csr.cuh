#ifndef _SPMV_CSR_H_
#define _SPMV_CSR_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "compute_utils.cuh"

template <typename T>
eval_t spmv_CSR_CPU(param_t params, CSR_Matrix<T> *mat, const T *x, T *y);

template <typename TIN, typename TOUT>
eval_t spmv_CSR(param_t params, CSR_Matrix<TIN> *mat, const TIN *x, TOUT *y);

eval_t spmv_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *xD, const float *xS, double *y);
eval_t spmv_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat, const double *xD,
                                    const float *xS, double *y);
eval_t spmv_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat, const double *xD,
                                   const float *xS, double *y);
eval_t spmv_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                        const double *xD, const float *xS, double *y);

#include "spmv_csr.tpp"

#endif  // _SPMV_CSR_H_