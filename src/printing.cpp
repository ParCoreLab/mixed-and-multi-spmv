#include "printing.hpp"

void print_parameters(param_t params) {
  if (!params.is_script) {
    printf("info Parameters:\n");
    printf("  Split Range: %lf\n", params.split_range);
    printf("  Split Percentage: %lf\n", params.split_percentage);
    printf("  Split Shrink Factor: %lf\n", params.split_shrink_factor);
    printf("  Split Option: %d\n", params.run_option);
    printf("  Is Square?: %s\n", params.is_square ? "Yes" : "No");
    printf("  Is Symmetric?: %s\n", params.is_symmetric ? "Yes" : "No");
    printf("  Empty rows: %d\n", params.empty_row_count);
    printf("  Avg. nz/row: %lf\n", params.avg_nz_inrow);
    printf("  Matrix Type: %c\n", params.mattype);
  }
}

void print_header(param_t params, compute_type_e computetype) {
  if (!params.is_script) {
    if (computetype == SPMV or computetype == CARDIAC) {
      printf("------------------------------------------------\n");
      printf("info %s computations with %d iterations.\n", computetype == SPMV ? "SpMV" : "Cardiac",
             computetype == SPMV ? params.spmv_iters : params.cardiac_iters);
      printf("\n");
      printf("%*s", PRNTSPC, "Name");
      printf("%*s", PRNTSPC, "Time (ms)");
      printf("%*s", PRNTSPC, "Speedup");
      printf("%*s", PRNTSPC, "Error");
      printf("%*s", PRNTSPC, "Doubles");
      printf("%*s", PRNTSPC, "Singles");
      printf("%*s", PRNTSPC, "Percentage");
      printf("\n");
    } else if (computetype == JACOBI) {
      printf("------------------------------------------------\n");
      printf("info Jacobi computations with %d iterations\n", params.jacobi_iters);
      printf("\n");
      printf("%*s", PRNTSPC, "Name");
      printf("%*s", PRNTSPC, "Time (ms)");
      printf("%*s", PRNTSPC, "Speedup");
      printf("%*s", PRNTSPC, "Error");
      printf("%*s", PRNTSPC, "Delta");
      printf("%*s", PRNTSPC, "Doubles");
      printf("%*s", PRNTSPC, "Singles");
      printf("%*s", PRNTSPC, "Percentage");
      printf("\n");
    } else if (computetype == TEST) {
      printf("warn No tests implemented.\n");
    }
  }
}

void print_evaluation(param_t params, eval_t evals, const char *title) {
  if (evals.type == SPMV or evals.type == CARDIAC) {
    if (params.is_script) {
      // scrp spmv || title || time || speedup || error || doubles || singles || percentage
      printf(evals.type == SPMV ? "scrp spmv" : "scrp cardiac");
      printf("||%s", title);
      printf("||%lf", evals.time_taken_millsecs);
      printf("||%lf", params.doublecusp_time / evals.time_taken_millsecs);
      printf("||%.5e", evals.error);
      printf("||%d", evals.doubleCount);
      printf("||%d", evals.singleCount);
      printf("||%.5f", evals.percentage);
      printf("\n");
    } else {
      printf("%*s", PRNTSPC, title);
      printf("%*lf", PRNTSPC, evals.time_taken_millsecs);
      printf("%*lf", PRNTSPC, params.doublecusp_time / evals.time_taken_millsecs);
      printf("%*.5e", PRNTSPC, evals.error);
      printf("%*d", PRNTSPC, evals.doubleCount);
      printf("%*d", PRNTSPC, evals.singleCount);
      printf("%*.5f", PRNTSPC, evals.percentage);
      printf("\n");
    }
  } else if (evals.type == JACOBI) {
    if (params.is_script) {
      // scrp jacobi || title || time || speedup || error || gamma || iters || doubles || singles || percentage
      printf("scrp jacobi");
      printf("||%s", title);
      printf("||%lf", evals.time_taken_millsecs);
      printf("||%lf", params.doublecusp_time / evals.time_taken_millsecs);
      printf("||%.5e", evals.error);
      printf("||%.5e", evals.delta);
      printf("||%d", evals.doubleCount);
      printf("||%d", evals.singleCount);
      printf("||%.5f", evals.percentage);
      printf("\n");
    } else {
      printf("%*s", PRNTSPC, title);
      printf("%*lf", PRNTSPC, evals.time_taken_millsecs);
      printf("%*f", PRNTSPC, params.doublecusp_time / evals.time_taken_millsecs);
      printf("%*.5e", PRNTSPC, evals.error);
      printf("%*.5e", PRNTSPC, evals.delta);
      printf("%*d", PRNTSPC, evals.doubleCount);
      printf("%*d", PRNTSPC, evals.singleCount);
      printf("%*.5f", PRNTSPC, evals.percentage);
      printf("\n");
    }
  } else {
    printf("error Unknown eval type: %d.\n", evals.type);
    assert(false);
  }
}

void print_run_info(param_t params, CSR_Matrix<double> *mat) {
  if (params.is_script) {
    printf("scrp info");                          // 0
    printf("||%s", params.matrix_path);           // 1
    printf("||%d", mat->M);                       // 2
    printf("||%d", mat->N);                       // 3
    printf("||%d", mat->nz);                      // 4
    printf("||%d", params.spmv_iters);            // 5
    printf("||%d", params.jacobi_iters);          // 6
    printf("||%d", params.cardiac_iters);         // 7
    printf("||%d", params.is_symmetric);          // 8
    printf("||%d", params.run_option);            // 9
    printf("||%d", params.empty_row_count);       // 10
    printf("||%lf", params.avg_nz_inrow);         // 11
    printf("||%d", params.min_nz_inrow);          // 12
    printf("||%d", params.max_nz_inrow);          // 13
    printf("||%lf", params.split_range);          // 14
    printf("||%lf", params.split_percentage);     // 15
    printf("||%lf", params.split_shrink_factor);  // 16
    printf("||%c", params.mattype);               // 17
    printf("\n");
  }
}

void print_COO(COO_Matrix *mat, const char *title, const char *fmt) {
  printf("%s:\n", title);
  for (int i = 0; i < mat->nz; ++i) {
    printf("%d\t%d\t", mat->rows[i], mat->cols[i]);
    printf(fmt, mat->vals[i]);
    printf("\n");
  }
  printf("\n");
}