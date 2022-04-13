#include "spmv_ellr.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Entrywise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_ELLR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
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
  ELLR_Matrix<float> *matS = (ELLR_Matrix<float> *)malloc(sizeof(ELLR_Matrix<float>));
  ELLR_Matrix<double> *matD = (ELLR_Matrix<double> *)malloc(sizeof(ELLR_Matrix<double>));
  split_entrywise(mat, precisions, matS, matD);
  nzD = matD->nz;
  nzS = matS->nz;
  // print_ELLR<double>(matD, "ELLR FP64 matrix", "%lf ");
  // print_ELLR<float>(matS, "ELLR FP32 matrix", "%f ");

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->M * matD->R * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->M * matD->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, matD->M * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->M * matS->R * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->M * matS->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, matS->M * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->M * matD->R * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->M * matD->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowlen, matD->M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->M * matS->R * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->M * matS->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowlen, matS->M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_ELLR_vector_Mixed_Entrywise_Split<2>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_ELLR_vector_Mixed_Entrywise_Split<4>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_ELLR_vector_Mixed_Entrywise_Split<8>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_ELLR_vector_Mixed_Entrywise_Split<16>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_ELLR_vector_Mixed_Entrywise_Split<32>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV ELLR MIXED ENTRYWISE SPLIT");
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
  free_ELLR<float>(matS);
  free_ELLR<double>(matD);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_ELLR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
                                     const double *xD, const float *xS, double *y) {
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
  ELLR_Matrix<float> *matS = (ELLR_Matrix<float> *)malloc(sizeof(ELLR_Matrix<float>));
  ELLR_Matrix<double> *matD = (ELLR_Matrix<double> *)malloc(sizeof(ELLR_Matrix<double>));
  split_rowwise(mat, precisions, matS, matD, PERMUTE_ROWS);
  sep = matS->M;  // after this many rows, double matrix starts
  nzS = matS->nz;
  nzD = matD->nz;
  mE = mat->M - matD->M - matS->M;  // #empty rows

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->M * matD->R * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->M * matD->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, matD->M * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->M * matS->R * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->M * matS->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, matS->M * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->M * matD->R * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->M * matD->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowlen, matD->M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->M * matS->R * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->M * matS->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowlen, matS->M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Split<2>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M - mE, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Split<4>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M - mE, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Split<8>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M - mE, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Split<16>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M - mE, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Split<32>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M - mE, matS->R, d_AS, d_xS, d_colsS, d_rowsS, matD->R, d_AD, d_xD,
                                             d_colsD, d_rowsD, d_y, sep, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ROWWISE SPLIT");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  double *tmp = (double *)malloc(mat->M * sizeof(double));
  permute_vector<double>(y, mat->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat->M * sizeof(double));                               // copy back

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
  free_ELLR<float>(matS);
  free_ELLR<double>(matD);
  free(tmp);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_ELLR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat,
                                         const double *xD, const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result

  /** HOST VARS **/
  float timems;                 // for timers
  int nzD, nzS, mE;             // for eval
  int iter;                     // for loops
  ELLR_Matrix<double> *mat_RO;  // clustered matrix
  int sep;                      // separator = the first row where precision is double
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);

  /** REORDERING **/
  mat_RO = (ELLR_Matrix<double> *)malloc(sizeof(ELLR_Matrix<double>));
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_ROWS);
  float *valsS = (float *)malloc(mat_RO->M * mat_RO->R * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS, mat_RO->M * mat_RO->R);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->M * mat_RO->R * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->M * mat_RO->R * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->M * mat_RO->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->M * mat_RO->R * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat_RO->M * mat_RO->R * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->M * mat_RO->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowlen, (mat_RO->M - mE) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat_RO->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat_RO->M - mE, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, mat_RO->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, mat_RO->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, mat_RO->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, mat_RO->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, mat_RO->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, SPMV_IS_ACCUMULATIVE);
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
  free_ELLR<double>(mat_RO);
  free(valsS);

  return create_spmv_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Dual
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t spmv_ELLR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, ELLR_Matrix<double> *mat, const double *xD,
                                    const float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result
  bool *d_rUseSingle;    // device ptr precision info

  /** HOST VARS **/
  float timems;                                                  // for timer
  int iter;                                                      // for loops
  int nzD, nzS, mE;                                              // for eval
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);  // TODO
  // NOTE: mE should not be used here!

  /** HOST SETUP **/
  float *valsS = (float *)malloc(mat->M * mat->R * sizeof(float));
  transfer_vector<double, float>(mat->vals, valsS, mat->M * mat->R);
  bool *rUseSingle = (bool *)malloc(mat->M * sizeof(bool));
  for (int i = 0; i < mat->M; i++) rUseSingle[i] = (precisions[i] != DOUBLE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat->M * mat->R * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat->M * mat->R * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->M * mat->R * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, mat->M * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rUseSingle, mat->M * sizeof(bool)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat->vals, mat->M * mat->R * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat->M * mat->R * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->M * mat->R * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowlen, mat->M * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rUseSingle, rUseSingle, mat->M * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual<2><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, mat->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual<4><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, mat->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual<8><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, mat->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual<16><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, mat->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual<32><<<numBlocks, THREADS_PER_BLOCK>>>(
          mat->M, mat->R, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, SPMV_IS_ACCUMULATIVE);
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
