#include "compute_drives.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Jacobi
//
// First, x is set to [1/n, 2/n, ..., n/n] and then y = Ax is calcualted in double precision.
// After that, x is reset to be all zeros, and we try to solve the system.
//
// We calculate relative residual || y - Ax' ||_2
// We also calculate || x - x' ||
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_jacobi_CSR(param_t params, CSR_Matrix<double> *matD, double *diagD, float *xS, double *xD, double *yD) {
  params.split_range = calculate_range(params, matD);
  assert(matD->M == matD->N);
  eval_t evals;                                                // evaluation object
  double *xTrue = (double *)malloc(matD->M * sizeof(double));  // groundtruth x
  jacobi_reset_vector(xTrue, matD->M);                         // sets values to [1/n, 2/n, ...]
  bool steps[3];
  // if (matD->M < MAX_ALLOWED_ROWS_FOR_PRINTING) print_vector<double>(xTrue, matD->M, "True X");

  // Find the y vector with the original matrix A as y <- Ax
  spmv_CSR_FP64_with_Diagonal(matD, diagD, xTrue, yD);

  // FP64 Jacobi
  evals = jacobi_CSR_Double(params, matD, diagD, xD, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->N);
  params.doublecusp_time = evals.time_taken_millsecs;
  print_evaluation(params, evals, ANAME_DOUBLES_CUSP);
  if (check_bad_convergence(evals)) {
    if (params.is_script)
      printf("scrp skip||%.5e\n", evals.error);
    else
      printf("warn Jacobi is bad, skipping... (e %.5e)\n", evals.error);
    free(xTrue);
    return;
  }
  // if we are only doing FP64 Jacobi, you can skip the rest
  if (params.do_fp64_jacobi_only) {
    free(xTrue);
    return;
  }

  // FP32 Jacobi with FP64 Reduction
  CSR_Matrix<float> *matS = duplicate_CSR<double, float>(matD);
  evals = jacobi_CSR_Single(params, matD, matS, diagD, xD, xS, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->N);
  print_evaluation(params, evals, ANAME_SINGLES_DR_CUSP);
  free_CSR<float>(matS);

  precision_e *p_entrywise = (precision_e *)malloc(matD->nz * sizeof(precision_e));  // allocate entrywise precisions

  // Entrywise Split - Baseline
  precisions_set_datadriven(p_entrywise, matD, 0.01, 0.0, ENTRYWISE);
  evals = jacobi_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, diagD, xD, xS, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->N);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE);

  // Entrywise Split
  precisions_set_datadriven(p_entrywise, matD, params.split_range, 0.0, ENTRYWISE);
  evals = jacobi_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, diagD, xD, xS, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->N);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT);

  free(p_entrywise);                                                              // free entrywise precisions
  precision_e *p_rowwise = (precision_e *)malloc(matD->M * sizeof(precision_e));  // allocate rowwise precisions
  precisions_set_datadriven(p_rowwise, matD, params.split_range, params.split_percentage,
                            ROWWISE);  // abs + %

  // Rowwise Split
  evals = jacobi_CSR_Mixed_Rowwise_Split(params, p_rowwise, matD, diagD, xD, xS, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT);

  // Rowwise Composite
  evals = jacobi_CSR_Mixed_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD);
  evals.delta = L2Norm<double>(xTrue, xD, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE);

  // Rowwise Composite | MIXED to DOUBLE
  SETSTEPS(steps, 0, 1, 1);
  evals = jacobi_CSR_Multi_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD, steps);
  evals.delta = L2Norm<double>(xTrue, xD, matD->M);
  print_evaluation(params, evals, ANAME_MULTI_DD_R_DOUBLE);

  // Rowwise Composite | SINGLES to MIXED to DOUBLE
  SETSTEPS(steps, 1, 1, 1);
  evals = jacobi_CSR_Multi_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD, steps);
  evals.delta = L2Norm<double>(xTrue, xD, matD->M);
  print_evaluation(params, evals, ANAME_MULTI_SINGLE_DD_R_DOUBLE);

  free(p_rowwise);  // free rowwise precisions
  free(xTrue);      // free truth array
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SpMV
//
// The resulting output vector is compared to the output vector of FP64 CUSP SpMV.
//
// We will not do symmetric permutation on rowwise split and rowwise clustered methods for our SpMV evaluations.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_spmv_CSR(param_t params, CSR_Matrix<double> *matD, float *xS, double *xD, double *yD) {
  params.split_range = calculate_range(params, matD);
  eval_t evals;  // evaluation object
  if (params.vector_path) {
    // read the vector
    double *vec = read_array(params.vector_path, matD->N);  // initial x read as array
    if (!vec) {
      printf("scrp error||could not read the dense vector (path: %s).\n", params.vector_path);
      return;
    }
    memcpy(xD, vec, matD->M * sizeof(double));
    free(vec);
    transfer_vector<double, float>(xD, xS, matD->N);
  } else {
    // generate your own vector
    write_vector_random<double>(xD, matD->N, SPMV_INITIAL_X_RANGE_LOWER, SPMV_INITIAL_X_RANGE_UPPER);
    transfer_vector<double, float>(xD, xS, matD->N);
  }
  double *truth = (double *)malloc(matD->M * sizeof(double));  // groundtruth

  // FP64 SpMV
  evals = spmv_CSR<double, double>(params, matD, xD, yD);
  evals.error = 0;
  memcpy(truth, yD, matD->M * sizeof(double));
  params.doublecusp_time = evals.time_taken_millsecs;
  print_evaluation(params, evals, ANAME_DOUBLES_CUSP);

  precision_e *p_entrywise = (precision_e *)malloc(matD->nz * sizeof(precision_e));  // entrywise precisions

  // Entrywise - Baseline
  precisions_set_datadriven(p_entrywise, matD, 1.0, 0.0, ENTRYWISE);
  evals = spmv_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE);

  // Entrywise
  precisions_set_datadriven(p_entrywise, matD, params.split_range, 0.0, ENTRYWISE);
  evals = spmv_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT);

  free(p_entrywise);                                                              // free entrywise precisions
  precision_e *p_rowwise = (precision_e *)malloc(matD->M * sizeof(precision_e));  // allocate rowwise precisions
  precisions_set_datadriven(p_rowwise, matD, params.split_range, params.split_percentage, ROWWISE);  // abs + %

  // Rowwise Split - AbsMean and Percentage Margin
  evals = spmv_CSR_Mixed_Rowwise_Split(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT);

  // Rowwise Dual - AbsMean and Percentage Margin
  evals = spmv_CSR_Mixed_Rowwise_Dual(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL);

  // Rowwise Composite - AbsMean and Percentage Margin
  evals = spmv_CSR_Mixed_Rowwise_Composite(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE);

  free(p_rowwise);  // free rowwise precisions

  // allocations for FP32
  float *yS = (float *)malloc(matD->M * sizeof(float));          // fp32 copy of the output
  CSR_Matrix<float> *matS = duplicate_CSR<double, float>(matD);  // fp32 copy of the matrix

  // FP32 SpMV with Single Reduction
  evals = spmv_CSR<float, float>(params, matS, xS, yS);
  transfer_vector<float, double>(yS, yD, matS->M);
  evals.error = L2Norm<double>(yD, truth, matS->M);
  print_evaluation(params, evals, ANAME_SINGLES_SR_CUSP);

  // FP32 SpMV with Double Reduction
  evals = spmv_CSR<float, double>(params, matS, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matS->M);
  print_evaluation(params, evals, ANAME_SINGLES_DR_CUSP);

  // frees for FP32
  free_CSR<float>(matS);  // free fp32 copy of the matrix
  free(yS);               // free fp32 copy of the output

  // frees
  free(truth);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cardiac Simulation by Dr. James D. Trotter & Prof. Xing Cai from Simula
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_cardiac_CSR(param_t params, CSR_Matrix<double> *matD, double *diagD, float *xS, double *xD, double *yD) {
  params.split_range = calculate_range(params, matD);
  assert(matD->M == matD->N);  // assert that matrix is square
  eval_t evals;                // evaluation object
  bool steps[3];
  if (params.vector_path) {
    // read the vector
    double *vec = read_array(params.vector_path, matD->N);  // initial x read as array
    if (!vec) {
      printf("scrp error||could not read the dense vector (path: %s).\n", params.vector_path);
      return;
    }
    memcpy(xD, vec, matD->M * sizeof(double));
    free(vec);
    transfer_vector<double, float>(xD, xS, matD->N);
  } else {
    // generate your own vector
    write_vector<double>(xD, matD->N, CARDIAC_INITIAL_X_VALUE);
    write_vector<float>(xS, matD->N, CARDIAC_INITIAL_X_VALUE);
  }
  double *truth = (double *)malloc(matD->M * sizeof(double));  // groundtruth

  // FP64 Cardiac
  evals = cardiac_CSR_Double(params, matD, diagD, xD, yD);
  evals.error = 0;
  memcpy(truth, yD, matD->M * sizeof(double));
  print_evaluation(params, evals, ANAME_DOUBLES_CUSP);

  // allocations for FP32
  CSR_Matrix<float> *matS = duplicate_CSR<double, float>(matD);  // fp32 copy of the matrix

  // FP32 SpMV with Double Reduction
  evals = cardiac_CSR_Single(params, matS, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matS->M);
  print_evaluation(params, evals, ANAME_SINGLES_DR_CUSP);

  // frees for FP32
  free_CSR<float>(matS);  // free fp32 copy of the matrix

  precision_e *p_entrywise = (precision_e *)malloc(matD->nz * sizeof(precision_e));  // entrywise precisions

  // Entrywise - Baseline
  precisions_set_datadriven(p_entrywise, matD, 1e-5, 0.0, ENTRYWISE);
  evals = cardiac_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE);

  // Entrywise
  precisions_set_datadriven(p_entrywise, matD, params.split_range, 0.0, ENTRYWISE);
  evals = cardiac_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT);

  free(p_entrywise);                                                              // free entrywise precisions
  precision_e *p_rowwise = (precision_e *)malloc(matD->M * sizeof(precision_e));  // allocate rowwise precisions
  precisions_set_datadriven(p_rowwise, matD, params.split_range, params.split_percentage, ROWWISE);  // abs + %

  // Rowwise Split - AbsMean and Percentage Margin
  evals = cardiac_CSR_Mixed_Rowwise_Split(params, p_rowwise, matD, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT);

  // Rowwise Dual - AbsMean and Percentage Margin
  evals = cardiac_CSR_Mixed_Rowwise_Dual(params, p_rowwise, matD, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL);

  // Rowwise Composite - AbsMean and Percentage Margin
  evals = cardiac_CSR_Mixed_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE);

  // Rowwise Composite | MIXED to DOUBLE
  SETSTEPS(steps, 0, 1, 1);
  evals = cardiac_CSR_Multi_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD, steps);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_MULTI_DD_R_DOUBLE);

  // Rowwise Composite | SINGLES to MIXED to DOUBLE
  SETSTEPS(steps, 1, 1, 1);
  evals = cardiac_CSR_Multi_Rowwise_Composite(params, p_rowwise, matD, diagD, xD, xS, yD, steps);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_MULTI_SINGLE_DD_R_DOUBLE);

  free(p_rowwise);  // free rowwise precisions

  // frees
  free(truth);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Profiling SpMV
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_profiling_CSR(param_t params, CSR_Matrix<double> *matD, float *xS, double *xD, double *yD) {
  params.split_range = calculate_range(params, matD);
  eval_t evals;                                                // evaluation object
  double *truth = (double *)malloc(matD->M * sizeof(double));  // groundtruth
  write_vector_random<double>(xD, matD->N, SPMV_INITIAL_X_RANGE_LOWER, SPMV_INITIAL_X_RANGE_UPPER);  // set x
  transfer_vector<double, float>(xD, xS, matD->N);  // create fp32 copy of x
  params.spmv_iters = 1;

  /// FP64 CUSP
  evals = spmv_CSR<double>(params, matD, xD, yD);
  evals.error = 0;
  memcpy(truth, yD, matD->M * sizeof(double));
  params.doublecusp_time = evals.time_taken_millsecs;
  print_evaluation(params, evals, ANAME_DOUBLES_CUSP);

  // Entrywise - Baseline
  precision_e *p_entrywise = (precision_e *)malloc(matD->nz * sizeof(precision_e));  // entrywise precisions
  precisions_set_datadriven(p_entrywise, matD, 1.0, 0.0, ENTRYWISE);
  evals = spmv_CSR_Mixed_Entrywise_Split(params, p_entrywise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE);
  free(p_entrywise);

  // Rowwise Split - AbsMean and Percentage Margin
  precision_e *p_rowwise = (precision_e *)malloc(matD->M * sizeof(precision_e));  // allocate rowwise precisions
  precisions_set_datadriven(p_rowwise, matD, params.split_range, params.split_percentage, ROWWISE);  // abs + %
  evals = spmv_CSR_Mixed_Rowwise_Split(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT);
  free(p_rowwise);

  // frees
  free(truth);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SpMV (for ELLPACK-R format)
//
// The resulting output vector is compared to the output vector of FP64 CUSP SpMV.
//
// We will not do symmetric permutation on rowwise split and rowwise clustered methods for our SpMV evaluations.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_spmv_ELLR(param_t params, ELLR_Matrix<double> *matD, float *xS, double *xD, double *yD) {
  eval_t evals;  // evaluation object
  if (params.vector_path) {
    // read the vector
    double *vec = read_array(params.vector_path, matD->N);  // initial x read as array
    if (!vec) {
      printf("scrp error||could not read the dense vector (path: %s).\n", params.vector_path);
      return;
    }
    memcpy(xD, vec, matD->M * sizeof(double));
    free(vec);
    transfer_vector<double, float>(xD, xS, matD->N);
  } else {
    // generate your own vector
    write_vector_random<double>(xD, matD->N, SPMV_INITIAL_X_RANGE_LOWER, SPMV_INITIAL_X_RANGE_UPPER);
    transfer_vector<double, float>(xD, xS, matD->N);
  }
  double *truth = (double *)malloc(matD->M * sizeof(double));  // groundtruth

  // FP64 SpMV
  evals = spmv_ELLR<double, double>(params, matD, xD, yD);
  evals.error = 0;
  memcpy(truth, yD, matD->M * sizeof(double));
  params.doublecusp_time = evals.time_taken_millsecs;
  print_evaluation(params, evals, ANAME_DOUBLES_CUSP);

  precision_e *p_entrywise = (precision_e *)malloc(matD->nz * sizeof(precision_e));  // entrywise precisions

  // Entrywise - Baseline
  precisions_set_datadriven(p_entrywise, matD, 1.0, 0.0, ENTRYWISE);
  evals = spmv_ELLR_Mixed_Entrywise_Split(params, p_entrywise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE);

  // Entrywise
  precisions_set_datadriven(p_entrywise, matD, params.split_range, 0.0, ENTRYWISE);
  evals = spmv_ELLR_Mixed_Entrywise_Split(params, p_entrywise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT);

  free(p_entrywise);  // free entrywise precisions

  precision_e *p_rowwise = (precision_e *)malloc(matD->M * sizeof(precision_e));  // allocate rowwise precisions
  precisions_set_datadriven(p_rowwise, matD, params.split_range, params.split_percentage, ROWWISE);  // abs + %

  // Rowwise Split - AbsMean and Percentage Margin
  evals = spmv_ELLR_Mixed_Rowwise_Split(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT);

  // Rowwise Dual - AbsMean and Percentage Margin
  evals = spmv_ELLR_Mixed_Rowwise_Dual(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL);

  // Rowwise Composite - AbsMean and Percentage Margin
  evals = spmv_ELLR_Mixed_Rowwise_Composite(params, p_rowwise, matD, xD, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matD->M);
  print_evaluation(params, evals, ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE);

  free(p_rowwise);  // free rowwise precisions

  // allocations for FP32
  float *yS = (float *)malloc(matD->M * sizeof(float));            // fp32 copy of the output
  ELLR_Matrix<float> *matS = duplicate_ELLR<double, float>(matD);  // fp32 copy of the matrix

  // FP32 SpMV with Single Reduction
  evals = spmv_ELLR<float, float>(params, matS, xS, yS);
  transfer_vector<float, double>(yS, yD, matS->M);
  evals.error = L2Norm<double>(yD, truth, matS->M);
  print_evaluation(params, evals, ANAME_SINGLES_SR_CUSP);

  // FP32 SpMV with Double Reduction
  evals = spmv_ELLR<float, double>(params, matS, xS, yD);
  evals.error = L2Norm<double>(yD, truth, matS->M);
  print_evaluation(params, evals, ANAME_SINGLES_DR_CUSP);

  // frees for FP32
  free_ELLR<float>(matS);  // free fp32 copy of the matrix
  free(yS);                // free fp32 copy of the output

  free(truth);
}