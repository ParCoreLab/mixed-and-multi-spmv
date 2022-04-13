#ifndef _SPMV_ELLR_H_
#define _SPMV_ELLR_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "compute_utils.cuh"

template <typename T>
eval_t spmv_ELLR_CPU(param_t params, ELLR_Matrix<T> *mat, const T *x, T *y);

template <typename TIN, typename TOUT>
eval_t spmv_ELLR(param_t params, ELLR_Matrix<TIN> *mat, const TIN *x, TOUT *y);

eval_t spmv_ELLR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
                                       const double *xD, const float *xS, double *y);
eval_t spmv_ELLR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
                                     const double *xD, const float *xS, double *y);
eval_t spmv_ELLR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat, const double *xD,
                                    const float *xS, double *y);
eval_t spmv_ELLR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
                                         const double *xD, const float *xS, double *y);

#include "spmv_ellr.tpp"

#endif  // _SPMV_ELLR_H_