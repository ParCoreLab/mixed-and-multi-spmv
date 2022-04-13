#ifndef _CARDIAC_H_
#define _CARDIAC_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "compute_utils.cuh"

eval_t cardiac_CSR_Double_Swapping(param_t params, CSR_Matrix<double> *mat, double *x, double *y);

eval_t cardiac_CSR_Double(param_t params, CSR_Matrix<double> *mat, double *diag, double *x, double *y);

eval_t cardiac_CSR_Single(param_t params, CSR_Matrix<float> *mat, const double *diag, double *xD, float *xS, double *y);

eval_t cardiac_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                         const double *diag, double *xD, float *xS, double *y);

eval_t cardiac_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                       const double *diag, double *xD, float *xS, double *y);

eval_t cardiac_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *diag, double *xD, float *xS, double *y);

eval_t cardiac_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                           const double *diag, double *xD, float *xS, double *y);

eval_t cardiac_CSR_Multi_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                           const double *diag, double *xD, float *xS, double *y, const bool steps[3]);

#endif  // _CARDIAC_H_