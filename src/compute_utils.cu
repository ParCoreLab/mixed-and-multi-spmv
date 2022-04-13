#include "compute_utils.cuh"

/* works correctly if argument is in +/-[2**-15, 2**17), or zero, infinity, NaN
 */
// https://forums.developer.nvidia.com/t/fast-float-to-double-conversion/39102
// @deprecated
//__device__ __forceinline__ double fast_float2double(float a) {
//  unsigned int ia = __float_as_int(a);
//  return __hiloint2double((((ia >> 3) ^ ia) & 0x07ffffff) ^ ia, ia << 29);
//}

size_t spmv_grid_size(int rows, double avg_nz_inrow) {
  // max blocks in cuda is 65535
  if (avg_nz_inrow <= 2) {
    return fmin(NUM_BLOCKS_MAX, ceil(rows / ((float)THREADS_PER_BLOCK / 2.0)));
  } else if (avg_nz_inrow <= 4) {
    return fmin(NUM_BLOCKS_MAX, ceil(rows / ((float)THREADS_PER_BLOCK / 4.0)));
  } else if (avg_nz_inrow <= 8) {
    return fmin(NUM_BLOCKS_MAX, ceil(rows / ((float)THREADS_PER_BLOCK / 8.0)));
  } else if (avg_nz_inrow <= 16) {
    return fmin(NUM_BLOCKS_MAX, ceil(rows / ((float)THREADS_PER_BLOCK / 16.0)));
  } else
    return fmin(NUM_BLOCKS_MAX, ceil(rows / ((float)THREADS_PER_BLOCK / 32.0)));

  // const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  // fmin(NUM_BLOCKS_MAX, (mat->M + (VECTORS_PER_BLOCK - 1)) /
  // VECTORS_PER_BLOCK); // K.Ahmad
}

size_t vector_kernels_grid_size(int len) { return fmin(NUM_BLOCKS_MAX, ceil((float)len / (float)THREADS_PER_BLOCK)); }

eval_t create_spmv_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds) {
  eval_t evals;
  evals.type = SPMV;
  evals.percentage = double(nzS) / double(nzD + nzS);
  evals.singleCount = nzS;
  evals.doubleCount = nzD;
  evals.iterations = iter;
  evals.time_taken_millsecs = milliseconds;
  evals.error = -999;  // dummy
  return evals;
}

eval_t create_cardiac_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds) {
  eval_t evals;
  evals.type = CARDIAC;
  evals.percentage = double(nzS) / double(nzD + nzS);
  evals.singleCount = nzS;
  evals.doubleCount = nzD;
  evals.iterations = iter;
  evals.time_taken_millsecs = milliseconds;
  evals.error = -999;  // dummy
  return evals;
}

eval_t create_jacobi_evaluation(int rows, int nzS, int nzD, int iter, float milliseconds, double delta, double error,
                                bool isConverged) {
  eval_t evals;
  evals.type = JACOBI;
  evals.percentage = double(nzS) / double(nzD + nzS);
  evals.singleCount = nzS;
  evals.doubleCount = nzD;
  evals.error = error;
  evals.iterations = isConverged ? iter : iter - 1;  // +1 is due to for loop when max iter is reached
  evals.time_taken_millsecs = milliseconds;
  evals.isConverged = isConverged;
  evals.delta = -999;  // dummy
  return evals;
}

void split_entrywise(CSR_Matrix<double> *mat, precision_e *p, CSR_Matrix<float> *matS, CSR_Matrix<double> *matD) {
  int nzS = 0, nzD = 0, i, j;

  // First, count the non-zeros that so you can allocate accordingly
  for (i = 0; i < mat->M; ++i)
    for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j)
      if (p[j] == SINGLE)
        nzS++;
      else if (p[j] == DOUBLE)
        nzD++;
  assert(nzS + nzD == mat->nz);

  // Allocations
  matS->M = mat->M;
  matS->N = mat->N;
  matS->nz = nzS;
  matS->rowptr = (int *)malloc((mat->M + 1) * sizeof(int));
  matS->cols = (int *)malloc(nzS * sizeof(int));
  matS->vals = (float *)malloc(nzS * sizeof(float));

  matD->M = mat->M;
  matD->N = mat->N;
  matD->nz = nzD;
  matD->rowptr = (int *)malloc((mat->M + 1) * sizeof(int));
  matD->cols = (int *)malloc(nzD * sizeof(int));
  matD->vals = (double *)malloc(nzD * sizeof(double));

  // Assigning the non-zeros
  int nzS_i = 0, nzD_i = 0;
  matS->rowptr[0] = 0;
  matD->rowptr[0] = 0;
  for (i = 0; i < mat->M; ++i) {
    matS->rowptr[i + 1] = matS->rowptr[i];
    matD->rowptr[i + 1] = matD->rowptr[i];
    for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
      if (p[j] == SINGLE) {
        matS->cols[nzS_i] = mat->cols[j];
        matS->vals[nzS_i] = (float)(mat->vals[j]);
        matS->rowptr[i + 1]++;
        nzS_i++;
      } else if (p[j] == DOUBLE) {
        matD->cols[nzD_i] = mat->cols[j];
        matD->vals[nzD_i] = mat->vals[j];
        matD->rowptr[i + 1]++;
        nzD_i++;
      }
    }
  }
  assert((nzS_i == nzS) && (nzD_i == nzD));
}

void split_entrywise(ELLR_Matrix<double> *mat, precision_e *p, ELLR_Matrix<float> *matS, ELLR_Matrix<double> *matD) {
  int nzS = 0, nzD = 0, RS = 0, RD = 0, jS_i = 0, jD_i = 0;

  // Early allocations
  matS->M = mat->M;
  matS->N = mat->N;
  matS->rowlen = (int *)malloc(mat->M * sizeof(int));
  matD->M = mat->M;
  matD->N = mat->N;
  matD->rowlen = (int *)malloc(mat->M * sizeof(int));

  // Count the row-lengths so you can allocate accordingly
  int nz_i = 0;
  for (int i = 0; i < mat->M; ++i) {
    // find number of elements in the row
    matS->rowlen[i] = 0;
    matD->rowlen[i] = 0;
    for (int j = mat->R * i; j < mat->R * i + mat->rowlen[i]; ++j) {
      if (p[nz_i] == SINGLE) {
        matS->rowlen[i]++;
        nzS++;
      } else if (p[nz_i] == DOUBLE) {
        matD->rowlen[i]++;
        nzD++;
      }
      nz_i++;  // iterates precisions
    }
    // store max
    if (matS->rowlen[i] > RS) RS = matS->rowlen[i];
    if (matD->rowlen[i] > RD) RD = matD->rowlen[i];
  }
  assert(nzS + nzD == mat->nz);

  // Computed allocations
  matS->R = RS;
  matS->nz = nzS;
  matS->cols = (int *)malloc(mat->M * RS * sizeof(int));
  matS->vals = (float *)malloc(mat->M * RS * sizeof(float));
  matD->R = RD;
  matD->nz = nzD;
  matD->cols = (int *)malloc(mat->M * RD * sizeof(int));
  matD->vals = (double *)malloc(mat->M * RD * sizeof(double));

  // assign values
  jS_i = 0;
  jD_i = 0;
  nz_i = 0;
  for (int i = 0; i < mat->M; ++i) {
    assert(jS_i == RS * i);
    assert(jD_i == RD * i);
    // assign non-zeros
    for (int j = mat->R * i; j < mat->R * i + mat->rowlen[i]; ++j) {
      if (p[nz_i] == SINGLE) {
        matS->cols[jS_i] = mat->cols[j];
        matS->vals[jS_i] = (float)(mat->vals[j]);
        jS_i++;
      } else if (p[nz_i] == DOUBLE) {
        matD->cols[jD_i] = mat->cols[j];
        matD->vals[jD_i] = mat->vals[j];
        jD_i++;
      }
      nz_i++;
    }
    // assign rest of the singles
    while (jS_i < RS * (i + 1)) {
      matS->cols[jS_i] = -1;
      matS->vals[jS_i] = 0;
      jS_i++;
    }
    // assign rest of the doubles
    while (jD_i < RD * (i + 1)) {
      matD->cols[jD_i] = -1;
      matD->vals[jD_i] = 0;
      jD_i++;
    }
  }
  assert((jS_i == matS->M * matS->R) && (jD_i == matD->M * matD->R));
}

void split_rowwise(CSR_Matrix<double> *mat, precision_e *p, CSR_Matrix<float> *matS, CSR_Matrix<double> *matD,
                   perm_type_matrix_t perm_type) {
  int nzS = 0;  // single nz count
  int nzD = 0;  // double nz count
  int mS = 0;   // single row count
  int mD = 0;   // double row count
  int mE = 0;   // empty row count
  int i, j, i_cur, nzS_i, nzD_i;

  // First, count the non-zeros that so you can allocate accordingly
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      nzS += mat->rowptr[i + 1] - mat->rowptr[i];
      mS++;
    } else if (p[i] == DOUBLE) {
      nzD += mat->rowptr[i + 1] - mat->rowptr[i];
      mD++;
    } else if (p[i] == EMPTY) {
      mE++;
    }
  }

  assert(nzS + nzD == mat->nz);
  assert(mS + mD + mE == mat->M);
  // printf("ROWS S:%d\tD: %d\t E: %d\n", mS, mD, mE);

  // Allocations
  matS->M = mS;
  matS->N = mat->N;
  matS->nz = nzS;
  matS->rowptr = (int *)malloc((mS + 1) * sizeof(int));
  matS->cols = (int *)malloc(nzS * sizeof(int));
  matS->vals = (float *)malloc(nzS * sizeof(float));

  matD->M = mD;
  matD->N = mat->N;
  matD->nz = nzD;
  matD->rowptr = (int *)malloc((mD + 1) * sizeof(int));
  matD->cols = (int *)malloc(nzD * sizeof(int));
  matD->vals = (double *)malloc(nzD * sizeof(double));

  // Singles
  matS->rowptr[0] = 0;
  i_cur = 0;
  nzS_i = 0;
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      matS->rowptr[i_cur + 1] = matS->rowptr[i_cur];
      for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
        matS->cols[nzS_i] = mat->cols[j];
        matS->vals[nzS_i] = mat->vals[j];
        matS->rowptr[i_cur + 1]++;
        nzS_i++;
      }
      i_cur++;
    }
  }

  // Doubles
  matD->rowptr[0] = 0;
  i_cur = 0;
  nzD_i = 0;
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == DOUBLE) {
      matD->rowptr[i_cur + 1] = matD->rowptr[i_cur];
      for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
        matD->cols[nzD_i] = mat->cols[j];
        matD->vals[nzD_i] = mat->vals[j];
        matD->rowptr[i_cur + 1]++;
        nzD_i++;
      }
      i_cur++;
    }
  }

  // empty rows are ignored

  // If symmetric, we need to permute the columns too.
  if (perm_type == PERMUTE_SYMMETRIC) {
    int *perm = (int *)malloc(mat->M * sizeof(int));
    get_permutations(p, mat->M, perm);

    // permute
    for (i = 0; i < matD->M; ++i)
      for (j = matD->rowptr[i]; j < matD->rowptr[i + 1]; ++j) matD->cols[j] = perm[matD->cols[j]];
    for (i = 0; i < matS->M; ++i)
      for (j = matS->rowptr[i]; j < matS->rowptr[i + 1]; ++j) matS->cols[j] = perm[matS->cols[j]];

    free(perm);
  }

  assert((nzS_i == nzS) && (nzD_i == nzD));
}

void split_rowwise(ELLR_Matrix<double> *mat, precision_e *p, ELLR_Matrix<float> *matS, ELLR_Matrix<double> *matD,
                   perm_type_matrix_t perm_type) {
  int nzS = 0, nzD = 0, RS = 0, RD = 0, iS = 0, iD = 0, mE = 0;

  // count the row-lengths so you can allocate accordingly
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      iS++;
    } else if (p[i] == DOUBLE) {
      iD++;
    } else if (p[i] == EMPTY) {
      mE++;
    }
  }
  assert(iS + iD + mE == mat->M);

  matS->M = iS;
  matS->N = mat->N;
  matS->rowlen = (int *)malloc(iS * sizeof(int));
  matD->M = iD;
  matD->N = mat->N;
  matD->rowlen = (int *)malloc(iD * sizeof(int));

  // populate row lengths
  iS = 0;
  iD = 0;
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      // single row
      matS->rowlen[iS] = mat->rowlen[i];
      nzS += mat->rowlen[i];
      // store max
      if (matS->rowlen[iS] > RS) RS = matS->rowlen[iS];
      iS++;
    } else if (p[i] == DOUBLE) {
      // double row
      matD->rowlen[iD] = mat->rowlen[i];
      nzD += mat->rowlen[i];
      // store max
      if (matD->rowlen[iD] > RD) RD = matD->rowlen[iD];
      iD++;
    }
  }
  assert(nzS + nzD == mat->nz);
  assert(iS == matS->M && iD == matD->M);

  matS->R = RS;
  matS->nz = nzS;
  matS->cols = (int *)malloc(matS->M * RS * sizeof(int));
  matS->vals = (float *)malloc(matS->M * RS * sizeof(float));
  matD->R = RD;
  matD->nz = nzD;
  matD->cols = (int *)malloc(matD->M * RD * sizeof(int));
  matD->vals = (double *)malloc(matD->M * RD * sizeof(double));

  // assigning the non-zeros
  int jS_i = 0, jD_i = 0;
  iS = 0;
  iD = 0;
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      for (int j = mat->R * i; j < mat->R * i + mat->rowlen[i]; ++j) {
        matS->cols[jS_i] = mat->cols[j];
        matS->vals[jS_i] = (float)(mat->vals[j]);
        jS_i++;
      }
      // assign rest of the singles
      while (jS_i < RS * (iS + 1)) {
        matS->cols[jS_i] = -1;
        matS->vals[jS_i] = 0;
        jS_i++;
      }
      iS++;
    } else if (p[i] == DOUBLE) {
      for (int j = mat->R * i; j < mat->R * i + mat->rowlen[i]; ++j) {
        matD->cols[jD_i] = mat->cols[j];
        matD->vals[jD_i] = mat->vals[j];
        jD_i++;
      }
      // assign rest of the doubles
      while (jD_i < RD * (iD + 1)) {
        matD->cols[jD_i] = -1;
        matD->vals[jD_i] = 0;
        jD_i++;
      }
      iD++;
    }
  }
  assert((jS_i == matS->M * matS->R) && (jD_i == matD->M * matD->R));
}

void get_nonzero_counts_rowwise(CSR_Matrix<double> *mat, precision_e *p, int *nzD, int *nzS, int *mE) {
  *nzD = 0;
  *nzS = 0;
  *mE = 0;
  for (int i = 0; i < mat->M; i++) {
    if (p[i] == SINGLE)
      *nzS = *nzS + (mat->rowptr[i + 1] - mat->rowptr[i]);
    else if (p[i] == DOUBLE)
      *nzD = *nzD + (mat->rowptr[i + 1] - mat->rowptr[i]);
    else if (p[i] == EMPTY)
      *mE = *mE + 1;  // empty rows
  }
  assert(*(nzD) + *(nzS) == mat->nz);
}
void get_nonzero_counts_rowwise(ELLR_Matrix<double> *mat, precision_e *p, int *nzD, int *nzS, int *mE) {
  *nzD = 0;
  *nzS = 0;
  *mE = 0;
  for (int i = 0; i < mat->M; i++) {
    if (p[i] == SINGLE)
      *nzS = *nzS + (mat->rowlen[i]);
    else if (p[i] == DOUBLE)
      *nzD = *nzD + (mat->rowlen[i]);
    else if (p[i] == EMPTY)
      *mE = *mE + 1;  // empty rows
  }
  assert(*(nzD) + *(nzS) == mat->nz);
}

void get_permutations(const precision_e *p, const int n, int *ans) {
  int j = 0, i;
  // singles first
  for (i = 0; i < n; ++i) {
    if (p[i] == SINGLE) {
      ans[i] = j++;
    }
  }

  // doubles second
  for (i = 0; i < n; ++i) {
    if (p[i] == DOUBLE) {
      ans[i] = j++;
    }
  }

  // empty rows third
  for (i = 0; i < n; ++i) {
    if (p[i] == EMPTY) {
      ans[i] = j++;
    }
  }

  assert(j == n);
}

bool check_bad_convergence(eval_t evals) {
  return (evals.error > JACOBI_SKIP_THRESHOLD_ERROR ||  // skip high error
          isnan(evals.error) ||                         // skip nan error
          isinf(evals.error)                            // skip inf error
  );
}

void jacobi_reset_vector(double *arr, const int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (double(i) + 1.0) / double(size);
  }
}

void spmv_CSR_FP64_with_Diagonal(CSR_Matrix<double> *mat, const double *diag, double *x, double *y) {
  /** DEVICE VARS **/
  int *d_cols, *d_rows;  // device ptrs common
  double *d_xD, *d_AD;   // device ptrs double
  double *d_diag, *d_y;  // device rhs and diag

  /** DEVICE SETUP **/
  CUDA_CHECK_CALL(cudaMalloc(&d_AD, mat->nz * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_cols, mat->nz * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_rows, (mat->M + 1) * sizeof(int)));
  CUDA_CHECK_CALL(cudaMalloc(&d_diag, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_y, mat->M * sizeof(double)));
  CUDA_CHECK_CALL(cudaMalloc(&d_xD, mat->N * sizeof(double)));
  CUDA_CHECK_CALL(cudaMemcpy(d_AD, mat->vals, mat->nz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_rows, mat->rowptr, (mat->M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_diag, diag, mat->M * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_xD, x, mat->N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemset(d_y, 0, (mat->M) * sizeof(double)));

  /** KERNEL LAUNCH **/
  const double avg_nz_inrow = (double)mat->nz / (double)mat->M;
  const size_t numBlocksSpMV = spmv_grid_size(mat->M, avg_nz_inrow);
  const size_t numBlocksVec = vector_kernels_grid_size(mat->M);

  // Just one SpMV
  if (avg_nz_inrow <= 2) {
    kernel_spmv_CSR_vector_CUSP<double, double, 2>
        <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
  } else if (avg_nz_inrow <= 4) {
    kernel_spmv_CSR_vector_CUSP<double, double, 4>
        <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
  } else if (avg_nz_inrow <= 8) {
    kernel_spmv_CSR_vector_CUSP<double, double, 8>
        <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
  } else if (avg_nz_inrow <= 16) {
    kernel_spmv_CSR_vector_CUSP<double, double, 16>
        <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);
  } else
    kernel_spmv_CSR_vector_CUSP<double, double, 32>
        <<<numBlocksSpMV, THREADS_PER_BLOCK>>>(mat->M, d_AD, d_xD, d_cols, d_rows, d_y, SPMV_IS_ACCUMULATIVE);

  CUDA_GET_LAST_ERR("JACOBI CSR SPMV");

  // add diagonal to the result
  kernel_add_diag<<<numBlocksVec, THREADS_PER_BLOCK>>>(d_xD, d_diag, d_y, mat->M);
  CUDA_GET_LAST_ERR("JACOBI CSR ADD DIAG");

  // copy result back to host
  CUDA_CHECK_CALL(cudaMemcpy(y, d_y, mat->M * sizeof(double), cudaMemcpyDeviceToHost));

  /** WRAP-UP **/
  CUDA_CHECK_CALL(cudaFree(d_AD));
  CUDA_CHECK_CALL(cudaFree(d_rows));
  CUDA_CHECK_CALL(cudaFree(d_cols));
  CUDA_CHECK_CALL(cudaFree(d_xD));
  CUDA_CHECK_CALL(cudaFree(d_diag));
  CUDA_CHECK_CALL(cudaFree(d_y));
}

void find_checkpoints(const int iters, const bool steps[3], int checkpoints[2]) {
  const int d = steps[0] + steps[1] + steps[2];  // denominator
  if (d == 0) {
    checkpoints[0] = 0;
    checkpoints[1] = 0;
  } else {
    int cur = 0;
    for (int i = 0; i < 2; i++) {
      if (steps[i]) cur += iters / d;
      checkpoints[i] = cur;
    }
    // it is possible to eliminate cur here for a tiny optimization
  }
}

double report_potential_occupancy(void *kernel) {
  cudaDeviceProp prop;
  int numBlocks, activeWarps, maxWarps;
  double occupancy;

  // calcualte potential for device 0
  CUDA_CHECK_CALL(cudaGetDeviceProperties(&prop, 0));
  CUDA_CHECK_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, THREADS_PER_BLOCK, 0));
  activeWarps = numBlocks * THREADS_PER_BLOCK / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  occupancy = (double)activeWarps / maxWarps;

  return occupancy;
}

void show_kernel_occupancies() {
  printf("kernel_spmv_CSR_vector_CUSP POTENTIAL: %lf\n",
         report_potential_occupancy((void *)kernel_spmv_CSR_vector_CUSP<double, double, 32>));
  printf("kernel_spmv_CSR_vector_Mixed_Rowwise_Dual POTENTIAL: %lf\n",
         report_potential_occupancy((void *)kernel_spmv_CSR_vector_Mixed_Rowwise_Dual<32>));
  printf("kernel_spmv_CSR_vector_Mixed_Rowwise_Split POTENTIAL: %lf\n",
         report_potential_occupancy((void *)kernel_spmv_CSR_vector_Mixed_Rowwise_Split<32>));
  printf("kernel_spmv_CSR_vector_Mixed_Rowwise_Composite POTENTIAL: %lf\n",
         report_potential_occupancy((void *)kernel_spmv_CSR_vector_Mixed_Rowwise_Composite<32>));
  printf("kernel_spmv_CSR_vector_Mixed_Entrywise_Split POTENTIAL: %lf\n",
         report_potential_occupancy((void *)kernel_spmv_CSR_vector_Mixed_Entrywise_Split<32>));
}