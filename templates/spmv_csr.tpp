#include "spmv_csr.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU FP64
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
eval_t spmv_CSR_CPU(param_t params, CSR_Matrix<T> *mat, const T *x, T *y) {
  /** HOST VARS **/
  float timems;       // for timer
  int iter, i, j;     // for loops
  T t;                // temp var
  double start, end;  // timers

  /** KERNEL LAUNCH **/
  start = omp_get_wtime();
  for (iter = 1; iter <= params.spmv_iters; ++iter) {
    for (i = 0; i < mat->M; ++i) {
      t = (SPMV_IS_ACCUMULATIVE) ? y[i] : 0.0;
      for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
        t += mat->vals[j] * x[mat->cols[j]];
      }
      y[i] = t;
    }
  }
  end = omp_get_wtime();
  timems = (end - start) / 1000.0;

  return create_spmv_evaluation(mat->M, sizeof(float) == sizeof(T) ? mat->nz : 0,
                                sizeof(float) == sizeof(T) ? 0 : mat->nz, iter, timems);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU FP64
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TIN, typename TOUT>
eval_t spmv_CSR(param_t params, CSR_Matrix<TIN> *mat, TIN *x, TOUT *y) {
  /** DEVICE VARS **/
  TIN *d_x, *d_A;        // device ptrs
  TOUT *d_y;             // device ptrs
  int *d_cols, *d_rows;  // device ptrs

  /** HOST VARS **/
  float timems;  // for timer
  int iter;      // for loops

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_A, mat->nz * sizeof(TIN)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_x, mat->N * sizeof(TIN)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(TOUT)));
  CUDA_CHECK_CALL(cudaMemcpy(d_A, mat->vals, mat->nz * sizeof(TIN), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_x, x, mat->N * sizeof(TIN), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, SPMV_INITIAL_Y_VALUE, mat->M * sizeof(TOUT)));

  /** KERNEL LAUNCH **/
  const size_t numBlocks = spmv_grid_size(mat->M, params.avg_nz_inrow);
  START_TIMERS();
  for (iter = 1; iter <= params.spmv_iters; iter++) {
    if (params.avg_nz_inrow <= 2) {
      kernel_spmv_CSR_vector_CUSP<TIN, TOUT, 2>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 4) {
      kernel_spmv_CSR_vector_CUSP<TIN, TOUT, 4>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 8) {
      kernel_spmv_CSR_vector_CUSP<TIN, TOUT, 8>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
    } else if (params.avg_nz_inrow <= 16) {
      kernel_spmv_CSR_vector_CUSP<TIN, TOUT, 16>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
    } else
      kernel_spmv_CSR_vector_CUSP<TIN, TOUT, 32>
          <<<numBlocks, THREADS_PER_BLOCK>>>(mat->M, d_A, d_x, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
    CUDA_GET_LAST_ERR("SPMV CSR CUSP");
  }
  STOP_TIMERS(timems);

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(TOUT), cudaMemcpyDeviceToHost));
  CUDA_CHECK_CALL(cudaFree(d_A));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_y));
  CUDA_CHECK_CALL(cudaFree(d_x));

  return create_spmv_evaluation(mat->M, sizeof(float) == sizeof(TIN) ? mat->nz : 0,
                                sizeof(float) == sizeof(TIN) ? 0 : mat->nz, iter, timems);
}
