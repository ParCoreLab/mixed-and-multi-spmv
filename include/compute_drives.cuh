#ifndef _COMPUTE_DRIVES_H_
#define _COMPUTE_DRIVES_H_

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "cardiac.cuh"
#include "compute_utils.cuh"
#include "jacobi.cuh"
#include "matrix.hpp"
#include "parameters.hpp"
#include "precisions.hpp"
#include "printing.hpp"
#include "spmv_csr.cuh"
#include "spmv_ellr.cuh"
#include "verify.hpp"

#define MAX_ALLOWED_ROWS_FOR_PRINTING 20

// Doubles
#define ANAME_DOUBLES_CPU "FP64 (CPU)"
#define ANAME_DOUBLES_CUSP "FP64 (CUSP)"
#define ANAME_DOUBLES_MIXED_ENTRYWISE_SPLIT "FP64 (ES)"
#define ANAME_DOUBLES_MIXED_ROWWISE_SPLIT "FP64 (RS)"
#define ANAME_DOUBLES_MIXED_ROWWISE_DUAL "FP64 (RD)"
#define ANAME_DOUBLES_MIXED_ROWWISE_COMPOSITE "FP64 (RC)"
// Singles
#define ANAME_SINGLES_SR_CUSP "FP32-S (CUSP)"
#define ANAME_SINGLES_DR_CUSP "FP32-D (CUSP)"
#define ANAME_SINGLES_MIXED_ENTRYWISE_SPLIT "FP32 (ES)"
#define ANAME_SINGLES_MIXED_ROWWISE_SPLIT "FP32 (RS)"
#define ANAME_SINGLES_MIXED_ROWWISE_DUAL "FP32 (RD)"
#define ANAME_SINGLES_MIXED_ROWWISE_COMPOSITE "FP32 (RC)"
// Mixed
#define ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT "DD (ES)"
#define ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE "DD (ESBASE)"
#define ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL "DD (RD)"
#define ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT "DD (RS)"
#define ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE "DD (RC)"
// Multi-Step
#define ANAME_MULTI_SINGLE_DOUBLE "FP32-FP64"
#define ANAME_MULTI_SINGLE_DD_R "FP32-DD(RC)"
#define ANAME_MULTI_SINGLE_DD_E "FP32-DD(EC)"
#define ANAME_MULTI_DD_R_DOUBLE "DD(RC)-FP64"
#define ANAME_MULTI_DD_E_DOUBLE "DD(EC)-FP64"
#define ANAME_MULTI_SINGLE_DD_R_DOUBLE "FP32-DD(RC)-FP64"
#define ANAME_MULTI_SINGLE_DD_E_DOUBLE "FP32-DD(EC)-FP64"

/**
 * @brief Driver function for SpMV (CSR) computations.
 *
 * @param params Parameters
 * @param matD FP64 CSR Matrix
 * @param xS FP32 dense vector
 * @param xD FP64 dense vector
 * @param yD FP64 rhs vector
 */
void compute_spmv_CSR(param_t params, CSR_Matrix<double> *matD, float *xS, double *xD, double *yD);

/**
 * @brief Driver function for SpMV (ELLR) computations.
 *
 * @param params Parameters
 * @param matD FP64 ELLR Matrix
 * @param xS FP32 dense vector
 * @param xD FP64 dense vector
 * @param yD FP64 rhs vector
 */
void compute_spmv_ELLR(param_t params, ELLR_Matrix<double> *matD, float *xS, double *xD, double *yD);

/**
 * @brief Driver function for Jacobi method (CSR) computations.
 *
 * @param params Parameters
 * @param matD FP64 CSR Matrix without diagonal
 * @param diagD FP64 Diagonal as vector
 * @param xS FP32 dense vector
 * @param xD FP64 dense vector
 * @param yD FP64 rhs vector
 */
void compute_jacobi_CSR(param_t params, CSR_Matrix<double> *matD, double *diagD, float *xS, double *xD, double *yD);

/**
 * @brief Driver function for Cardiac simulations.
 *
 * @param params Parameters
 * @param matD FP64 CSR Matrix
 * @param xS FP32 dense vector
 * @param xD FP64 dense vector
 * @param yD FP64 rhs vector
 */
void compute_cardiac_CSR(param_t params, CSR_Matrix<double> *matD, double *diagD, float *xS, double *xD, double *yD);

/**
 * @brief Profile SpMV CSR kernels with nvprof.
 *
 * @param params Parameters
 * @param matD FP64 CSR Matrix
 * @param xS FP32 dense vector
 * @param xD FP64 dense vector
 * @param yD FP64 rhs vector
 */
void compute_profiling_CSR(param_t params, CSR_Matrix<double> *matD, float *xS, double *xD, double *yD);

#endif  // _COMPUTE_DRIVES_H_