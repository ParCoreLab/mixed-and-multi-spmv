#include <cuda.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compute_drives.cuh"
#include "matrix.hpp"
#include "parameters.hpp"
#include "printing.hpp"

param_t parse_command_line_args(int argc, char *argv[]);
void print_device_properties();

int main(int argc, char *argv[]) {
  // reset randomizer
  srand(time(NULL));

  param_t params = parse_command_line_args(argc, argv);
  if (params.matrix_path == NULL) {
    printf("scrp error||no matrix specified\n");
    exit(1);
  }
  if (!(params.do_cardiac || params.do_jacobi || params.do_spmv || params.do_spmv_ellr || params.do_profiling)) {
    printf("warn no computations selected! defaulting to spmv\n");
    params.do_spmv = true;
  }

  // check if there is a GPU
  int deviceCount;
  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess || deviceCount == 0) {
    printf("scrp error||no gpus\n");
    return 1;
  } else
    cudaSetDevice(0);

  // print device properties
  if (!params.is_script) print_device_properties();

  // check if there is a matrix at the given path
  if (!file_exists(params.matrix_path)) {
    printf("scrp error||matrix does not exist\n");
    return 1;
  }

  // read COO matrix
  printf("info Reading COO matrix from %s\n", params.matrix_path);
  COO_Matrix *coo = read_COO(params.matrix_path);
  if (!coo) {
    printf("scrp error||bad matrix");
    return 1;
  }
  printf("info COO Matrix ready (%d x %d) with %d nnz (type: %c).\n", coo->M, coo->N, coo->nz, coo->type);
  params.is_symmetric = coo->isSymmetric;
  params.is_square = (coo->M == coo->N);
  params.mattype = coo->type;

  // check if matrix is square (for jacobi)
  if (params.do_jacobi && !params.is_square) {
    printf("warn Matrix is not square. Will not do Jacobi.\n");
    params.do_jacobi = false;
    if (!params.do_spmv) {
      printf("scrp error||non-square in Jacobi only run.\n");
      free_COO(coo);
      return 1;
    }
  }
  if (params.do_jacobi) {
    // Jacobi Preprocess
    if (params.mattype != 'r') {
      printf("scrp error||bad matrix type: %c\n", params.mattype);
      free_COO(coo);
      return 1;
    }
    int zindiag, zoffdiag, numdiags;
    count_problematic_values(coo, &zindiag, &zoffdiag, &numdiags);
    /* explicit zeros off-diagonal are NOT problematic */
    ;
    /* zeros in diagonal cause problems for HSL */
    if (zindiag > 0) {
      printf("scrp error||%d zero(s) in diagonal\n", zindiag);
      free_COO(coo);
      return 1;
    }
    /* a matrix made up of only diagonal is bad*/
    if (numdiags == coo->nz) {
      printf("scrp error||matrix is only made of diagonal\n");
      free_COO(coo);
      return 1;
    }
    /* empty diagonals cause problems for HSL*/
    if (numdiags < coo->M) {
      printf("scrp error||matrix has %d empty diagonal(s)\n", coo->M - numdiags);
      free_COO(coo);
      return 1;
    }
    // permute and scale
    if (coo->M < MAX_ALLOWED_ROWS_FOR_PRINTING) print_COO(coo, "COO (before)");
    printf("info Running HSL MC64...\n");
    int retcode = hsl_permute_and_scale(coo);
    if (retcode != 0) {
      printf("scrp error||HSL gave error %d", retcode);
      free_COO(coo);
      return 1;
    }
    if (coo->M < MAX_ALLOWED_ROWS_FOR_PRINTING) print_COO(coo, "COO (after)");
  }
  // convert COO to CSR
  if (coo->isSymmetric) {
    printf("info Duplicating off-diagonal entries...\n");
    duplicate_off_diagonals(coo);
  }
  CSR_Matrix<double> *matOrig = COO_to_CSR(coo);
  if (matOrig) {
    printf("info CSR Matrix ready (%d x %d) with %d nnz.\n", matOrig->M, matOrig->N, matOrig->nz);
    free_COO(coo);
  } else {
    printf("scrp error||not enough memory during COO to CSR\n");
    free_COO(coo);
    return 1;
  }

  // count empty rows
  find_row_densities(matOrig, &params.min_nz_inrow, &params.max_nz_inrow, &params.avg_nz_inrow);
  params.empty_row_count = count_empty_rows(matOrig);
  print_parameters(params);
  print_run_info(params, matOrig);

  // start
  printf("info Resetting device...\n");
  CUDA_CHECK_CALL(cudaDeviceReset());
  /// create vectors
  float *xS = (float *)malloc(matOrig->N * sizeof(float));     // size is # cols
  double *xD = (double *)malloc(matOrig->N * sizeof(double));  // size is # cols
  double *yD = (double *)malloc(matOrig->M * sizeof(double));  // size is # rows
  if (xS == NULL || xD == NULL || yD == NULL) {
    if (xS) free(xS);
    if (xD) free(xD);
    if (yD) free(yD);
    printf("scrp error||could not allocate vectors\n");
    return 1;
  }
  if (params.do_profiling) {
    // Profiling (no warmup)
    print_header(params, SPMV);
    compute_profiling_CSR(params, matOrig, xS, xD, yD);
  } else {
    // Warmup
    param_t warmup_params = params;
    warmup_params.spmv_iters = 200;
    write_vector_random<double>(xD, matOrig->N, -5, 5);        // set dense vector
    spmv_CSR<double, double>(warmup_params, matOrig, xD, yD);  // do warm-up spmv FP64
    printf("info %d Iterations warm-up done.\n", 200);

    if (params.do_spmv) {  // SpMV (CSR)
      // compute
      print_header(params, SPMV);
      compute_spmv_CSR(params, matOrig, xS, xD, yD);
      free_CSR<double>(matOrig);
    } else if (params.do_jacobi) {  // Jacobi (CSR)
      CSR_Matrix<double> *matRem = create_CSR<double>(matOrig->M, matOrig->N, matOrig->nz - matOrig->M);
      double *diag = (double *)malloc(matOrig->M * sizeof(double));

      if (matRem && diag) {
        // extract diagonal
        extract_diagonal(matOrig, diag, matRem);
        free_CSR<double>(matOrig);
        // compute
        print_header(params, JACOBI);
        compute_jacobi_CSR(params, matRem, diag, xS, xD, yD);
        free_CSR<double>(matRem);
        free(diag);
      } else {
        free_CSR<double>(matOrig);
      }

    } else if (params.do_cardiac) {  // Cardiac (CSR)
      // find time with swapping first (no copy kernels)
      eval_t tmp = cardiac_CSR_Double_Swapping(params, matOrig, xD, yD);
      params.doublecusp_time = tmp.time_taken_millsecs;

      CSR_Matrix<double> *matRem = create_CSR<double>(matOrig->M, matOrig->N, matOrig->nz - matOrig->M);
      double *diag = (double *)malloc(matOrig->M * sizeof(double));
      if (matRem && diag) {
        // extract diagonal
        extract_diagonal(matOrig, diag, matRem);
        free_CSR<double>(matOrig);
        // compute
        print_header(params, CARDIAC);
        compute_cardiac_CSR(params, matRem, diag, xS, xD, yD);
        free_CSR<double>(matRem);
        free(diag);
      } else {
        free_CSR<double>(matOrig);
      }

    } else if (params.do_spmv_ellr) {
      params.split_range = calculate_range(params, matOrig);

      // CSR to ELLR
      ELLR_Matrix<double> *matELLR = CSR_to_ELLR<double, double>(matOrig);
      free_CSR<double>(matOrig);

      if (matELLR) {
        // compute
        print_header(params, SPMV);
        compute_spmv_ELLR(params, matELLR, xS, xD, yD);
        free_ELLR<double>(matELLR);
      }
    }
  }

  // free vectors
  free(xS);
  free(xD);
  free(yD);
  printf("\ninfo bye bye!\n");

  return 0;
}

param_t parse_command_line_args(int argc, char *argv[]) {
  param_t params;
  // set default parameters
  params.cardiac_iters = 500;
  params.spmv_iters = 500;
  params.jacobi_iters = 500;
  params.empty_row_count = 0;
  params.avg_nz_inrow = 0;
  params.min_nz_inrow = 0;
  params.max_nz_inrow = 0;
  params.split_range = 1.0;
  params.split_percentage = 99.0;
  params.split_range_hsl = 0.01;
  params.split_shrink_factor = 0.1;
  params.doublecusp_time = 1.0;
  params.is_script = false;
  params.is_square = false;
  params.is_symmetric = false;
  params.run_option = 1;
  params.do_fp64_jacobi_only = false;
  params.do_jacobi = false;
  params.do_spmv = false;
  params.do_spmv_ellr = false;
  params.do_profiling = false;
  params.do_cardiac = false;
  params.matrix_path = NULL;
  params.vector_path = NULL;
  params.mattype = 'x';
  // read user inputs
  for (int ac = 1; ac < argc; ac++) {
    ///////////////////////////////// RESOURCES ///////////////////////////////
    // matrix path
    if (MATCH_INPUT("-m")) {
      params.matrix_path = argv[++ac];
    }
    // vector path
    else if (MATCH_INPUT("-x")) {
      params.vector_path = argv[++ac];
    }
    ///////////////////////////////// SCRIPTED ////////////////////////////////
    // prints are in favor of scripts
    else if (MATCH_INPUT("-s")) {
      params.is_script = true;
    }
    ///////////////////////////////// PROFILING ///////////////////////////////
    // run profiling
    else if (MATCH_INPUT("--profile")) {
      params.do_profiling = true;
    }
    ///////////////////////////////// CARDIAC /////////////////////////////////
    // run cardiac simulation
    else if (MATCH_INPUT("--cardiac")) {
      params.do_cardiac = true;
    }
    // iteration count
    else if (MATCH_INPUT("--cardiac-i")) {
      params.cardiac_iters = atoi(argv[++ac]);
    }
    ///////////////////////////////// SPMV ////////////////////////////////////
    // run spmv
    else if (MATCH_INPUT("--spmv")) {
      params.do_spmv = true;
    }
    // iteration count
    else if (MATCH_INPUT("--spmv-i")) {
      params.spmv_iters = atoi(argv[++ac]);
    }
    // run ELLPACK-R spmv
    else if (MATCH_INPUT("--spmv-ellr")) {
      params.do_spmv_ellr = true;
    }
    ///////////////////////////////// JACOBI //////////////////////////////////
    // run jacobi
    else if (MATCH_INPUT("--jacobi")) {
      params.do_jacobi = true;
    }
    // iteration count
    else if (MATCH_INPUT("--jacobi-i")) {
      params.jacobi_iters = atoi(argv[++ac]);
    }
    // only do FP64 Jacobi to see who diverge or not
    else if (MATCH_INPUT("--jacobi-fp64")) {
      params.do_fp64_jacobi_only = true;
      params.do_jacobi = true;
    }
    ///////////////////////////////// SPLITTING ///////////////////////////////
    // datadriven range (-r, r) (for opt 0)
    else if (MATCH_INPUT("--split-r")) {
      params.split_range = atof(argv[++ac]);
    }
    // datadriven range (-r, r) after HSL (for opt 0)
    else if (MATCH_INPUT("--split-rh")) {
      params.split_range_hsl = atof(argv[++ac]);
    }
    // datadriven percentage p of in-range nnz's in a row
    else if (MATCH_INPUT("--split-p")) {
      params.split_percentage = atof(argv[++ac]);
    }
    // do max-scaling in the matrix, make sure you set the range too
    else if (MATCH_INPUT("--split-opt")) {
      params.run_option = atoi(argv[++ac]);
    }
    // shrink factor (for opt 1)
    else if (MATCH_INPUT("--split-s")) {
      params.split_shrink_factor = atof(argv[++ac]);
    }
    ///////////////////////////////// USAGE ///////////////////////////////////
    else {
      printf(
          "Usage: %s\n"
          "-m <path>       Input Matrix path.\n"
          "-x <path>       Input Vector path.\n"
          "-s              Run in script mode (for evaluator.py).\n"
          "--split-r <x>   Data-driven range (for option 0).\n"
          "--split-p <x>   Data-driven row-wise percentage.\n"
          "--split-rh <x>  Data-driven range after HSL. (for opt. 0)\n"
          "--split-s <x>   Shrink factor (for opt. 1)\n"
          "--profile       Do a profiling run.\n"
          "--cardiac       Do cardiac simulation.\n"
          "--cardiac-i <x> Do x iterations of Cardiac.\n"
          "--spmv          Do SpMV.\n"
          "--spmv-i <x>    Do x iterations of SpMV.\n"
          "--jacobi        Do Jacobi.\n"
          "--jacobi-i <x>  Do x iterations of Jacobi.\n"
          "--jacobi-fp64   Only do FP64 Jacobi.\n"
          "--opt <no>      Running option.\n",
          argv[0]);
      exit(1);
    }
  }
  return params;
}

void print_device_properties() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf(
      "info Using device %s (CC %d.%d)\n\t"
      "Max Threads per Block: %d\n\t"
      "Warp Size: %d threads\n\t"
      "Concurrent Kernels: %s\n\t"
      "Global Memory: %lf GB\n",
      prop.name, prop.major, prop.minor, prop.maxThreadsPerBlock, prop.warpSize,
      (prop.concurrentKernels == 0) ? "False" : "True", 1e-9 * (double)(prop.totalGlobalMem));
}
