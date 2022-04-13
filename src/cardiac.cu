#include "cardiac.cuh"

eval_t cardiac_CSR_Double_Swapping(param_t params, CSR_Matrix<double> *mat, double *x, double *y) {
  /** DEVICE VARS **/
  double *d_x, *d_A;     // device ptrs
  double *d_y;           // device ptrs
  int *d_cols, *d_rows;  // device ptrs
  double *d_tmp;         // device ptr temp for swapping

  /** HOST VARS **/
  float timems;  // for timer
  int iter;      // for loops

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_A, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_x, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_A, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_x, x, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<double, double, 2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<double, double, 4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<double, double, 8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<double, double, 16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else
      kernel_spmv_CSR_vector_CUSP<double, double, 32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR CUSP");

    // swap
    d_tmp = d_x;
    d_x = d_y;
    d_y = d_tmp;
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_x, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK_CALL(cudaFree(d_A));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_x));

  return create_cardiac_evaluation(mat->M, 0, mat->nz, iter, timems);
}

eval_t cardiac_CSR_Double(param_t params, CSR_Matrix<double> *mat, double *diag, double *x, double *y) {
  /** DEVICE VARS **/
  double *d_x, *d_A;     // device ptrs
  double *d_y;           // device ptrs
  int *d_cols, *d_rows;  // device ptrs
  double *d_diag;        // device ptr diagonal

  /** HOST VARS **/
  float timems;  // for timer
  int iter;      // for loops

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_A, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_x, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_A, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_x, x, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<double, double, 2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<double, double, 4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<double, double, 8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<double, double, 16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    } else
      kernel_spmv_CSR_vector_CUSP<double, double, 32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR CUSP");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_x, NULL, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_x, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK_CALL(cudaFree(d_A));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_x));

  return create_cardiac_evaluation(mat->M, 0, mat->nz, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU FP32 with FP64 reduction
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Single(param_t params, CSR_Matrix<float> *mat, const double *diag, double *xD, float *xS,
                          double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_A;     // device ptrs
  double *d_xD;          // device ptrs
  double *d_y;           // device ptrs
  int *d_cols, *d_rows;  // device ptrs
  double *d_diag;        // device ptr for diagonal

  /** HOST VARS **/
  float timems;  // for timer
  int iter;      // for loops

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_A, mat->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_A, mat->vals, mat->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<float, double, 2>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_xS, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<float, double, 4>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_xS, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<float, double, 8>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_xS, d_cols, d_rows, d_y, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<float, double, 16>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_xS, d_cols, d_rows, d_y, false);
    } else
      kernel_spmv_CSR_vector_CUSP<float, double, 32>
          <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_A, d_xS, d_cols, d_rows, d_y, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR CUSP");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK_CALL(cudaFree(d_A));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_xD));

  return create_cardiac_evaluation(mat->M, mat->nz, 0, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Entrywise Mixed
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Mixed_Entrywise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                         const double *diag, double *xD, float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_y;             // device ptr result
  double *d_diag;          // device ptr for diagonal

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
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();

  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, false);
    } else
      kernel_spmv_CSR_vector_Mixed_Entrywise_Split<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR MIXED ENTRYWISE SPLIT");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

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
  CUDA_CHECK_CALL(cudaFree(d_diag));
  free_CSR<float>(matS);
  free_CSR<double>(matD);

  return create_cardiac_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Mixed_Rowwise_Split(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                       const double *diag, double *xD, float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;      // device ptrs single
  int *d_colsS, *d_rowsS;  // device ptrs single
  double *d_xD, *d_AD;     // device ptrs double
  int *d_colsD, *d_rowsD;  // device ptrs double
  double *d_y;             // device ptr result
  double *d_diag;          // device ptr for diagonal

  /** HOST VARS **/
  float timems;      // for timer
  int iter;          // for loops
  int nzD, nzS, mE;  // for eval
  int sep;           // separator = the first row where precision is double
  double *tmp = (double *)malloc(mat->M * sizeof(double));

  /** REORDERING & SPLITTING **/
  CSR_Matrix<float> *matS = (CSR_Matrix<float> *)malloc(sizeof(CSR_Matrix<float>));
  CSR_Matrix<double> *matD = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  split_rowwise(mat, precisions, matS, matD, PERMUTE_SYMMETRIC);
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
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, matD->vals, matD->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsD, matD->cols, matD->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsD, matD->rowptr, (matD->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, matS->vals, matS->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_colsS, matS->cols, matS->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rowsS, matS->rowptr, (matS->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat->M, precisions, tmp, PERMUTE_FORWARD);  // permute diags
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(xD, mat->M, precisions, tmp, PERMUTE_FORWARD);  // permute xD
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, tmp, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  transfer_vector<double, float>(tmp, xS, mat->M);  // permute xS
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  mE = 0;
  const size_t numBlocksSpMV = spmv_grid_size(mat->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, false);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Split<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M - mE, d_AS, d_xS, d_colsS, d_rowsS, d_AD, d_xD, d_colsD, d_rowsD, d_y, sep, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR MIXED ROWWISE SPLIT");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  permute_vector<double>(y, mat->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat->M * sizeof(double));                               // copy back
  transfer_vector<double, float>(xD, xS, mat->M);                        // restore unpermuted xD to xS

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
  CUDA_CHECK_CALL(cudaFree(d_diag));
  free_CSR<float>(matS);
  free_CSR<double>(matD);
  free(tmp);

  return create_cardiac_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Mixed_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                           const double *diag, double *xD, float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result
  double *d_diag;        // device ptr for diagonal

  /** HOST VARS **/
  float timems;                // for timers
  int nzD, nzS, mE;            // for eval
  int iter;                    // for loops
  CSR_Matrix<double> *mat_RO;  // clustered matrix
  int sep;                     // separator = the first row where precision is double
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);

  /** REORDERING **/
  mat_RO = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_SYMMETRIC);
  double *tmp = (double *)malloc(mat_RO->M * sizeof(double));  // tmp vector
  float *valsS = (float *)malloc(mat_RO->nz * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS, mat_RO->nz);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M + 1 - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat_RO->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowptr, (mat_RO->M + 1 - mE) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);  // permute diags
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(xD, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);  // permute xD
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, tmp, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  transfer_vector<double, float>(tmp, xS, mat_RO->N);  // permute xS
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat_RO->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat_RO->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR MIXED ROWWISE DUAL CLUSTERED");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat_RO->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  permute_vector<double>(y, mat_RO->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat_RO->M * sizeof(double));                               // copy back
  transfer_vector<double, float>(xD, xS, mat->M);                           // restore unpermuted xD to xS

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<double>(mat_RO);
  free(valsS);
  free(tmp);

  return create_cardiac_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Mixed Rowwise Dual
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Mixed_Rowwise_Dual(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                      const double *diag, double *xD, float *xS, double *y) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result
  bool *d_rUseSingle;    // device ptr precision info
  double *d_diag;        // device ptr for diagonal

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
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rUseSingle, rUseSingle, mat->M * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, xD, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    // spmv
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, false);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, false);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, false);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, false);
    } else
      kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
          mat->M, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, d_rUseSingle, false);
    CUDA_GET_LAST_ERR("CARDIAC CSR MIXED ROWWISE DUAL");

    // update
    kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
    CUDA_GET_LAST_ERR("CARDIAC UPDATE");
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rUseSingle));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free(rUseSingle);
  free(valsS);

  return create_cardiac_evaluation(mat->M, nzS, nzD, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Multi Rowwise Composite
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
eval_t cardiac_CSR_Multi_Rowwise_Composite(param_t params, precision_e *precisions, CSR_Matrix<double> *mat,
                                           const double *diag, double *xD, float *xS, double *y, const bool steps[3]) {
  /** DEVICE VARS **/
  float *d_xS, *d_AS;    // device ptrs single
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_y;           // device ptr result
  double *d_diag;        // device ptr for diagonal

  /** HOST VARS **/
  float timems;                // for timers
  int nzD, nzS, mE;            // for eval
  int iter;                    // for loops
  CSR_Matrix<double> *mat_RO;  // clustered matrix
  int sep;                     // separator = the first row where precision is double
  get_nonzero_counts_rowwise(mat, precisions, &nzD, &nzS, &mE);
  int checkpoints[2];
  find_checkpoints(params.cardiac_iters, steps, checkpoints);

  /** REORDERING **/
  mat_RO = (CSR_Matrix<double> *)malloc(sizeof(CSR_Matrix<double>));
  permute_matrix<double>(mat, precisions, mat_RO, &sep, PERMUTE_SYMMETRIC);
  double *tmp = (double *)malloc(mat_RO->M * sizeof(double));  // tmp vector
  float *valsS = (float *)malloc(mat_RO->nz * sizeof(float));
  transfer_vector<double, float>(mat_RO->vals, valsS, mat_RO->nz);

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat_RO->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_AS, mat_RO->nz * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat_RO->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat_RO->M + 1 - mE) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat_RO->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xS, mat_RO->N * sizeof(float)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat_RO->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat_RO->vals, mat_RO->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_AS, valsS, mat_RO->nz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat_RO->cols, mat_RO->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat_RO->rowptr, (mat_RO->M + 1 - mE) * sizeof(int), cudaMemcpyHostToDevice));
  permute_vector<double>(diag, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);  // permute diags
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, tmp, mat_RO->M * sizeof(double), cudaMemcpyHostToDevice));
  permute_vector<double>(xD, mat_RO->M, precisions, tmp, PERMUTE_FORWARD);  // permute xD
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, tmp, mat_RO->N * sizeof(double), cudaMemcpyHostToDevice));
  transfer_vector<double, float>(tmp, xS, mat_RO->M);  // permute xS
  CUDA_CHECK_CALL(cudaMemcpy(d_xS, xS, mat_RO->N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0.0, mat_RO->M * sizeof(double)));

  /** KERNEL LAUNCH **/
  const size_t numBlocksSpMV = spmv_grid_size(mat_RO->M - mE, params.avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);
  START_TIMERS();
  for (iter = 1; iter <= params.cardiac_iters; iter++) {
    if (iter <= checkpoints[0]) {
      // fp32 spmv with double reduction
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_CUSP<float, double, 2>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_CUSP<float, double, 4>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_CUSP<float, double, 8>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_CUSP<float, double, 16>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_cols, d_rows, d_y, false);
      } else
        kernel_spmv_CSR_vector_CUSP<float, double, 32>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AS, d_xS, d_cols, d_rows, d_y, false);
      CUDA_GET_LAST_ERR("CARDIAC CSR CUSP");

      // update
      kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
      CUDA_GET_LAST_ERR("CARDIAC UPDATE");
    } else if (iter <= checkpoints[1]) {
      // mixed row-wise composite spmv
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<2><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
            mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<4><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
            mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<8><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
            mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<16><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
            mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
      } else
        kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32><<<numBlocksSpMV, THREADS_PER_BLOCK>>>(
            mat_RO->M - mE, d_AS, d_xS, d_AD, d_xD, d_cols, d_rows, d_y, sep, false);
      CUDA_GET_LAST_ERR("CARDIAC CSR MIXED ROWWISE DUAL CLUSTERED");

      // update
      kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, d_xS, mat->M);
      CUDA_GET_LAST_ERR("CARDIAC UPDATE");
    } else {
      // fp64 spmv
      if (params.avg_nz_inrow <= 2) {
        kernel_spmv_CSR_vector_CUSP<double, double, 2>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 4) {
        kernel_spmv_CSR_vector_CUSP<double, double, 4>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 8) {
        kernel_spmv_CSR_vector_CUSP<double, double, 8>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, false);
      } else if (params.avg_nz_inrow <= 16) {
        kernel_spmv_CSR_vector_CUSP<double, double, 16>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, false);
      } else
        kernel_spmv_CSR_vector_CUSP<double, double, 32>
            <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, false);
      CUDA_GET_LAST_ERR("CARDIAC CSR CUSP");

      // update
      kernel_cardiac_update<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_y, d_diag, d_xD, NULL, mat->M);
      CUDA_GET_LAST_ERR("CARDIAC UPDATE");
    }
  }
  STOP_TIMERS(timems);
  CUDA_CHECK_CALL(cudaMemcpy(y, d_xD, mat_RO->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** REVERSE REORDERING **/
  permute_vector<double>(y, mat_RO->M, precisions, tmp, PERMUTE_BACKWARD);  // clustered result
  memcpy(y, tmp, mat_RO->M * sizeof(double));                               // copy back
  transfer_vector<double, float>(xD, xS, mat_RO->M);                        // restore unpermuted xD to xS

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AS));
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_xS));
  CUDA_CHECK_CALL(cudaFree(d_y));
  free_CSR<double>(mat_RO);
  free(valsS);
  free(tmp);

  return create_cardiac_evaluation(mat->M, nzS, nzD, iter, timems);
}