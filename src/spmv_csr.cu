#include "spmv_csr.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Entrywise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *xD, const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_y;             // device ptr result

  /** HOST VARS **/
  float timems;  // for timer
  int iter;      // for loops
  int nzD, nzS;  // for eval

  /** SPLITTING **/
  CSR_Matrix<float> *matS = (CSR_Matrix<float> *)malloc(sizeof(CSR_Matrix<float>));
  CSR_Matrix<double> *matD = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  split_entrywise(mat, precisions, matS, matD);
  nzD = matD->nz;
  nzS = matS->nz;

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, (matD->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, (matS->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ENTRYWISE SPLIT");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_colsS));
  CUDA_CHECK_CALL(cudaFree(d_rowsS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_colsD));
  CUDA_CHECK_CALL(cudaFree(d_rowsD));
  free_CSR<float>(matS);
  free_CSR<double>(matD);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat, const double *xD,
                                    const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_y;             // device ptr result

  /** HOST VARS **/
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  int sep;           // separator = the first row where precision is double

  /** REORDERING & SPLITTING **/
  CSR_Matrix<float> *matS = (CSR_Matrix<float> *)malloc(sizeof(CSR_Matrix<float>));
  CSR_Matrix<double> *matD = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  split_rowwise(mat, precisions, matS, matD, PERMUTE_ROWS);
  sep = matS->M;  // after this many rows, double matrix starts
  nzS = matS->nz;
  nzD = matD->nz;
  mE = mat->M - matD->M - matS->M;  // #empty rows

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, (matD->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, (matS->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ROWWISE SPLIT");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  double *tmp = (double *)malloc(mat->M * sizeof(double));
  permute_vector<double>(y, mat->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat->M * sizeof(double));                               // copy back
  free(tmp);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_colsS));
  CUDA_CHECK_CALL(cudaFree(d_rowsS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_colsD));
  CUDA_CHECK_CALL(cudaFree(d_rowsD));
  free_CSR<float>(matS);
  free_CSR<double>(matD);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                        const double *xD, const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result

  /** HOST VARS **/
  float timems;                // for timers
  int nzD, nzS, mE;            // for eval
  int iter;                    // for loops
  CSR_Matrix<double> *mat_RO;  // clustered matrix
  int sep;                     // separator = the first row where precision is double
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);

  /** REORDERING **/
  mat_RO = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_ROWS);
  float *valsS = (float *)malloc(mat_RO->nz * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS, mat_RO->nz);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M + 1 - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat_RO->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowptr, (mat_RO->M + 1 - mE) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat_RO->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat_RO->M - mE, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ROWWISE DUAL CLUSTERED");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat_RO->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  double *tmp = (double *)malloc(mat_RO->M * sizeof(double));
  permute_vector<double>(y, mat_RO->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat_RO->M * sizeof(double));                               // copy back
  free(tmp);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<double>(mat_RO);
  free(valsS);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Dual
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat, const double *xD,
                                   const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result
  bool *d_rUseSingle;    // device ptr precision info

  /** HOST VARS **/
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);
  // NOTE: mE should not be used here!

  /** HOST SETUP **/
  float *valsS = (float *)malloc(mat->nz * sizeof(float));
  transfer_vector<double, float>(mat->vals, valsS, mat->nz);
  bool *rUseSingle = (bool *)malloc(mat->M * sizeof(bool));
  for (int i = 0; i < mat->M; i++) rUseSingle[i] = (precisions[i] != DOUBLE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rUseSingle, mat->M * sizeof(bool)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rUseSingle, rUseSingle, mat->M * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ROWWISE DUAL");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rUseSingle));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free(rUseSingle);
  free(valsS);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}
