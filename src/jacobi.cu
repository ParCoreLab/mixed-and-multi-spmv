#include "jacobi.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU FP64
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Double(param_t params, CSR_Matrix<double> *mat, const double *diag, double *x, const double *y) {
  /** DEVICE  VARS **/
  double *d_x, *d_A;     // device vector and matrix values
  double *d_xbar;        // device ptr temporary vector
  double *d_diag, *d_y;  // device rhs and diag
  int *d_cols, *d_rows;  // device row and col ptr

  /** HOST VARS **/
  bool isConverged = false;
  float timems;
  int iter;
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));

  /** HOST SETUP **/
  write_vector<double>(x, mat->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_A, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_x, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_A, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_y, y, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_x, x, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<double, double, 2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<double, double, 4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<double, double, 8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<double, double, 16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_xbar);
    } else
      kernel_spmv_CSR_vector_CUSP<double, double, 32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_xbar);
    CUDA_GET_LAST_ERR("JACOBI CSR CUSP");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_x, NULL, mat->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(x, d_x, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_A));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_x));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, x, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, 0, mat->nz, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU FP32
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Single(param_t params, CSR_Matrix<double> *matD, CSR_Matrix<float> *matS, const double *diag,
                         double *xD, float *xS, const double *y) {
  /** DEVICE  VARS **/
  float *d_xS, *d_AS;    // device vector and matrix values
  double *d_xD;          // device vector
  double *d_xbar;        // device ptr temporary vector
  double *d_diag, *d_y;  // device rhs and diag
  int *d_cols, *d_rows;  // device row and col ptr

  /** HOST VARS **/
  bool isConverged = false;
  float timems;
  int iter;
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(matS->M * sizeof(double));

  /** HOST SETUP **/
  write_vector<double>(xD, matS->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, matS->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, matS->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (matS->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, matS->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, matS->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, matS->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, matS->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, matS->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, matS->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_y, y, matS->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, matS->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, matS->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, matS->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(matS->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(matS->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<float, double, 2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(matS->M, d_AS, d_xS, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<float, double, 4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(matS->M, d_AS, d_xS, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<float, double, 8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(matS->M, d_AS, d_xS, d_cols, d_rows, d_xbar);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<float, double, 16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(matS->M, d_AS, d_xS, d_cols, d_rows, d_xbar);
    } else
      kernel_spmv_CSR_vector_CUSP<float, double, 32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(matS->M, d_AS, d_xS, d_cols, d_rows, d_xbar);
    CUDA_GET_LAST_ERR("JACOBI CSR CUSP");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, matS->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(xD, d_xD, matS->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(matD, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, matD->M);
  free(tmp);

  return create_jacobi_evaluation(matD->M, matD->nz, 0, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Entrywise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                        const double *diag, double *xD, float *xS, const double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_xbar;          // device ptr temporary vector
  double *d_diag, *d_y;    // device rhs and diag

  /** HOST VARS **/
  bool isConverged = false;
  float timems;  // for timer
  int iter;      // for loops
  int nzD, nzS;  // for eval
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));

  /** SPLITTING **/
  CSR_Matrix<float> *matS = (CSR_Matrix<float> *)malloc(sizeof(CSR_Matrix<float>));
  CSR_Matrix<double> *matD = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  split_entrywise(mat, precisions, matS, matD);
  nzD = matD->nz;
  nzS = matS->nz;

  /** HOST SETUP **/
  write_vector<double>(xD, mat->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, mat->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, (matD->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, (matS->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_y, y, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar);
    } else
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar);
    CUDA_GET_LAST_ERR("JACOBI CSR MIXED ENTRYWISE SPLIT");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(xD, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_colsS));
  CUDA_CHECK_CALL(cudaFree(d_rowsS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_colsD));
  CUDA_CHECK_CALL(cudaFree(d_rowsD));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<float>(matS);
  free_CSR<double>(matD);

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, nzS, nzD, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *diag, double *xD, float *xS, const double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_xbar;          // device ptr temporary vector
  double *d_diag, *d_y;    // device rhs and diag

  /** HOST VARS **/
  bool isConverged = false;
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));

  /** SPLITTING **/
  CSR_Matrix<float> *matS = (CSR_Matrix<float> *)malloc(sizeof(CSR_Matrix<float>));
  CSR_Matrix<double> *matD = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  split_rowwise(mat, precisions, matS, matD, PERMUTE_SYMMETRIC);
  nzD = matD->nz;
  nzS = matS->nz;
  mE = mat->M - matD->M - matS->M;

  /** REORDERING **/
  int sep = matS->M;  // after this many rows, double matrix starts

  /** HOST SETUP **/
  write_vector<double>(xD, mat->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, mat->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, matD->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsD, matD->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsD, (matD->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, matS->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_colsS, matS->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rowsS, (matS->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(y, mat->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_y, tmp, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar, sep);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_xbar, sep);
    CUDA_GET_LAST_ERR("JACOBI CSR MIXED ROWWISE SPLIT");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(tmp, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));
  permute_vector<double>(tmp, mat->M, precisions, xD, PERMUTE_BACKWARD);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_colsS));
  CUDA_CHECK_CALL(cudaFree(d_rowsS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_colsD));
  CUDA_CHECK_CALL(cudaFree(d_rowsD));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<float>(matS);
  free_CSR<double>(matD);

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, nzS, nzD, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                          const double *diag, double *xD, float *xS, const double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs single
  double *d_xD, *d_AD;   // device ptrs double
  double *d_xbar;        // device ptr temporary vector
  double *d_diag, *d_y;  // device rhs and diag

  /** HOST VARS **/
  bool isConverged = false;
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);

  /** REORDERING **/
  int sep;  // separator = the first row where precision is double
  CSR_Matrix<double> *mat_RO =
      (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));  // clustered matrix (can be in situ too)
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_SYMMETRIC);

  /** HOST SETUP **/
  write_vector<double>(xD, mat_RO->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, mat_RO->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M + 1 - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->nz * sizeof(double), cudaMemcpyHostToDevice));
  float *valsS_RO = (float *)malloc(mat_RO->nz * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS_RO, mat_RO->nz);
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS_RO, mat_RO->nz * sizeof(float), cudaMemcpyHostToDevice));
  free(valsS_RO);
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowptr, (mat_RO->M + 1 - mE) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat_RO->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(y, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_y, tmp, mat_RO->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
    CUDA_GET_LAST_ERR("JACOBI CSR MIXED ROWWISE SPLIT");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat_RO->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(tmp, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));
  permute_vector<double>(tmp, mat->M, precisions, xD, PERMUTE_BACKWARD);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<double>(mat_RO);

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, nzS, nzD, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Dual
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                     const double *diag, double *xD, float *xS, const double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_xbar;        // device ptr temporary vector
  double *d_diag, *d_y;  // device rhs and diag
  bool *d_rUseSingle;    // device ptr precision info

  /** HOST VARS **/
  bool isConverged = false;
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);
  // NOTE: mE should not be used here!

  /** HOST SETUP **/
  write_vector<double>(xD, mat->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, mat->N, JACOBI_INITIAL_X_VALUE);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rUseSingle, mat->M * sizeof(bool)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  float *valsS = (float *)malloc(mat->nz * sizeof(float));
  transfer_vector<double, float>(mat->vals, valsS, mat->nz);
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat->nz * sizeof(float), cudaMemcpyHostToDevice));
  free(valsS);
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  bool *rUseSingle = (bool *)malloc(mat->M * sizeof(bool));
  for (int i = 0; i < mat->M; i++) rUseSingle[i] = precisions[i] != DOUBLE;
  CUDA_CHECK_CALL(cudaMemcpy(d_rUseSingle, rUseSingle, mat->M * sizeof(bool), cudaMemcpyHostToDevice));
  free(rUseSingle);
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_y, y, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, d_rUseSingle);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, d_rUseSingle);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, d_rUseSingle);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, d_rUseSingle);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, d_rUseSingle);
    CUDA_GET_LAST_ERR("SPMV CSR MIXED ROWWISE DUAL");

    // jacobi iteration and solution update
    kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(xD, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rUseSingle));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_xbar));

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, nzS, nzD, iter, timems, delta, error, isConverged);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Multi Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t jacobi_CSR_Multi_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                          const double *diag, double *xD, float *xS, const double *y,
                                          const bool steps[3]) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs single
  double *d_xD, *d_AD;   // device ptrs double
  double *d_xbar;        // device ptr temporary vector
  double *d_diag, *d_y;  // device rhs and diag

  /** HOST VARS **/
  bool isConverged = false;
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  double delta = -1, error = NAN;
  double *tmp = (double *)malloc(mat->M * sizeof(double));
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);

  /** REORDERING **/
  int sep;  // separator = the first row where precision is double
  CSR_Matrix<double> *mat_RO =
      (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));  // clustered matrix (can be in situ too)
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_SYMMETRIC);

  /** HOST SETUP **/
  write_vector<double>(xD, mat_RO->N, JACOBI_INITIAL_X_VALUE);
  write_vector<float>(xS, mat_RO->N, JACOBI_INITIAL_X_VALUE);
  int checkpoints[2];
  find_checkpoints(params.jacobi_iters, steps, checkpoints);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M + 1 - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xbar, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->nz * sizeof(double), cudaMemcpyHostToDevice));
  float *valsS_RO = (float *)malloc(mat_RO->nz * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS_RO, mat_RO->nz);
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS_RO, mat_RO->nz * sizeof(float), cudaMemcpyHostToDevice));
  free(valsS_RO);
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowptr, (mat_RO->M + 1 - mE) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat_RO->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(y, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);
  CUDA_CHECK_CALL(cudaMemcpy(d_y, tmp, mat_RO->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_xbar, 0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.jacobi_iters; iter++) {
    if (iter <= checkpoints[0]) {
      // fp32 spmv with double reduction
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_CUSP<float, double, 2>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_CUSP<float, double, 4>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_CUSP<float, double, 8>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_CUSP<float, double, 16>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_cols, d_rows, d_xbar);
      } else
        kernel_spmv_CSR_vector_CUSP<float, double, 32>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_cols, d_rows, d_xbar);
      CUDA_GET_LAST_ERR("JACOBI CSR CUSP FP32");

      // jacobi iteration and solution update
      kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat_RO->M);
      CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
    } else if (iter <= checkpoints[1]) {
      // mixed row-wise composite spmv
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<2>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<4>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<8>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<16>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
      } else
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_xbar, sep);
      CUDA_GET_LAST_ERR("JACOBI CSR MIXED ROWWISE SPLIT");

      // jacobi iteration and solution update
      kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, d_xS, mat_RO->M);
      CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
    } else {
      // fp64 spmv
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_CUSP<double, double, 2>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AD, d_xD, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_CUSP<double, double, 4>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AD, d_xD, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_CUSP<double, double, 8>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AD, d_xD, d_cols, d_rows, d_xbar);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_CUSP<double, double, 16>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AD, d_xD, d_cols, d_rows, d_xbar);
      } else
        kernel_spmv_CSR_vector_CUSP<double, double, 32>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat_RO->M - mE, d_AD, d_xD, d_cols, d_rows, d_xbar);
      CUDA_GET_LAST_ERR("JACOBI CSR CUSP FP64");

      // jacobi iteration and solution update
      kernel_jacobi_iterate_and_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xbar, d_xD, NULL, mat_RO->M);
      CUDA_GET_LAST_ERR("JACOBI CSR ITERATE AND UPDATE");
    }
  }
  STOP_TIMERS(timems);

  /** RETREIVE RESULTS **/
  CUDA_CHECK_CALL(cudaMemcpy(tmp, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));
  permute_vector<double>(tmp, mat->M, precisions, xD, PERMUTE_BACKWARD);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xbar));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<double>(mat_RO);

  /** CALCULATE RESIDUAL **/
  spmv_CSR_FP64_with_Diagonal(mat, diag, xD, tmp);
  error = L2Norm<double>(tmp, y, mat->M);
  free(tmp);

  return create_jacobi_evaluation(mat->M, nzS, nzD, iter, timems, delta, error, isConverged);
}
