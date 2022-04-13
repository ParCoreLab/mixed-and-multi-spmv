#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>

#include "compute_utils.cuh"

eval_t jacobi_CSR_Double(param_t params, CSR_Matrix<double> *mat, const double *diag, double *x, const double *y);
eval_t jacobi_CSR_Single(param_t params, CSR_Matrix<double> *matD, CSR_Matrix<float> *matS, const double *diag,
                         double *xD, float *xS, const double *y);
eval_t jacobi_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                        const double *diag, double *xD, float *xS, const double *y);
eval_t jacobi_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *diag, double *xD, float *xS, const double *y);
eval_t jacobi_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                     const double *diag, double *xD, float *xS, const double *y);
eval_t jacobi_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                          const double *diag, double *xD, float *xS, const double *y);
eval_t jacobi_CSR_Multi_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                          const double *diag, double *xD, float *xS, const double *y,
                                          const bool steps[3]);

#endif  // _JACOBI_H_