#include "matrix.hpp"

void free_COO(COO_Matrix *mat) {
  free(mat->rows);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}

COO_Matrix *create_COO(int rows, int cols, int nonzeros) {
  COO_Matrix *mat = (COO_Matrix *)malloc(sizeof(COO_Matrix));
  mat->rows = (int *)calloc(nonzeros, sizeof(int));
  mat->cols = (int *)calloc(nonzeros, sizeof(int));
  mat->vals = (double *)calloc(nonzeros, sizeof(double));
  mat->M = rows;
  mat->N = cols;
  mat->nz = nonzeros;
  mat->isSymmetric = false;  // by default
  if (mat->rows == NULL || mat->cols == NULL || mat->vals == NULL) {
    if (mat->rows) free(mat->rows);
    if (mat->cols) free(mat->cols);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

COO_Matrix *read_COO(const char *path) {
  MM_typecode matcode;
  FILE *f;
  int i;

  if ((f = fopen(path, "r")) == NULL) {
    return NULL;
  }
  if (mm_read_banner(f, &matcode) != 0) {
    return NULL;
  }

  // Allow only certain matrices
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode))  // has to be real/int/binary
        && mm_is_coordinate(matcode)                                               // has to be in COO format
        && mm_is_sparse(matcode)                                                   // has to be sparse
        && !mm_is_dense(matcode)                                                   // can not be an array
        )) {
    return NULL;
  }

  // Obtain size info
  int M, N, nz;
  if ((mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
    return NULL;
  }

  int *rIndex = (int *)malloc(nz * sizeof(int));
  int *cIndex = (int *)malloc(nz * sizeof(int));
  double *val = (double *)malloc(nz * sizeof(double));

  /* When reading in floats, ANSI C requires the use of the "l"       */
  /* specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /* (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)           */
  /* also use %lg for reading int too */
  if (mm_is_real(matcode) || mm_is_integer(matcode)) {
    double tmp;
    for (i = 0; i < nz; i++) {
      fscanf(f, "%d %d %lg\n", &(rIndex[i]), &(cIndex[i]), &tmp);
      rIndex[i]--;  // 1-indexed to 0-indexed
      cIndex[i]--;  // 1-indexed to 0-indexed
      val[i] = tmp;
    }
  } else if (mm_is_pattern(matcode)) {
    for (i = 0; i < nz; i++) {
      fscanf(f, "%d %d\n", &(rIndex[i]), &(cIndex[i]));
      rIndex[i]--;  // 1-indexed to 0-indexed
      cIndex[i]--;  // 1-indexed to 0-indexed
      val[i] = 1.0;
    }
  } else
    return NULL;

  if (f != stdin) fclose(f);
  COO_Matrix *mat = (COO_Matrix *)malloc(sizeof(COO_Matrix));
  mat->M = M;
  mat->N = N;
  mat->nz = nz;
  mat->rows = rIndex;
  mat->cols = cIndex;
  mat->vals = val;
  mat->isSymmetric = mm_is_symmetric(matcode);
  if (mm_is_real(matcode)) mat->type = 'r';
  if (mm_is_integer(matcode)) mat->type = 'i';
  if (mm_is_pattern(matcode)) mat->type = 'p';
  return mat;
}

void extract_diagonal(CSR_Matrix<double> *mat, double *diag, CSR_Matrix<double> *matnew) {
  int numdiags = 0;
  int nz_i = 0;
  matnew->rowptr[0] = 0;
  for (int i = 0; i < mat->M; ++i) {
    matnew->rowptr[i + 1] = matnew->rowptr[i];
    for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
      // if (nz_i > matnew->nz) printf("NZ: %d\tM: %d\tI: %d\tJ: %d\tNZ_I: %d\n", matnew->nz, matnew->M, i, j, nz_i);
      if (i != mat->cols[j]) {
        // off-diagonal
        matnew->cols[nz_i] = mat->cols[j];
        matnew->vals[nz_i] = mat->vals[j];
        matnew->rowptr[i + 1]++;
        nz_i++;
      } else {
        // diagonal
        numdiags++;
        diag[i] = mat->vals[j];
      }
    }
  }
  assert(nz_i == matnew->nz);
  assert(numdiags == mat->M && mat->M == matnew->M);  // confirm we have seen each row
}

void duplicate_off_diagonals(COO_Matrix *mat) {
  // count number of off-diagonal entries
  int off_diagonals = 0, i;
  for (i = 0; i < mat->nz; i++)
    if (mat->rows[i] != mat->cols[i]) off_diagonals++;

  // allocate new memory for the actual matrix
  int true_nz = mat->nz + off_diagonals;  // 2 * off_diagonals + (nz - off_diagonals)
  int *new_rows = (int *)malloc(true_nz * sizeof(int));
  int *new_cols = (int *)malloc(true_nz * sizeof(int));
  double *new_vals = (double *)malloc(true_nz * sizeof(double));

  // populate the new values
  int new_i = 0;
  for (i = 0; i < mat->nz; i++) {
    // copy original
    new_rows[new_i] = mat->rows[i];
    new_cols[new_i] = mat->cols[i];
    new_vals[new_i] = mat->vals[i];
    new_i++;
    // if off diagonal, copy the symmetric value
    if (mat->rows[i] != mat->cols[i]) {
      new_cols[new_i] = mat->rows[i];  // row to col
      new_rows[new_i] = mat->cols[i];  // col to row
      new_vals[new_i] = mat->vals[i];  // but same val
      new_i++;
    }
  }

  // free old pointers
  free(mat->rows);
  free(mat->cols);
  free(mat->vals);
  // assign new pointers
  mat->rows = new_rows;
  mat->cols = new_cols;
  mat->vals = new_vals;
  // now the matrix is not symmetric (values are explicit)
  mat->isSymmetric = false;
  // update nz value
  mat->nz = true_nz;
  assert(new_i == true_nz);
}

CSR_Matrix<double> *COO_to_CSR(COO_Matrix *coo) {
  CSR_Matrix<double> *csr = create_CSR<double>(coo->M, coo->N, coo->nz);
  if (!csr) return NULL;

  int i;
  for (i = 0; i < coo->nz; i++) csr->rowptr[coo->rows[i] + 2]++;
  for (i = 2; i < coo->M + 2; i++) csr->rowptr[i] += csr->rowptr[i - 1];
  for (i = 0; i < coo->nz; i++) {
    csr->cols[csr->rowptr[coo->rows[i] + 1]] = coo->cols[i];
    csr->vals[csr->rowptr[coo->rows[i] + 1]++] = coo->vals[i];
  }
  assert(csr->rowptr[csr->M] == coo->nz);
  return csr;
}

CSC_Matrix *COO_to_CSC(COO_Matrix *coo) {
  CSC_Matrix *csc = create_CSC(coo->M, coo->N, coo->nz);
  if (!csc) return NULL;

  int i;
  for (i = 0; i < coo->nz; i++) csc->colptr[coo->cols[i] + 2]++;
  for (i = 2; i < coo->M + 2; i++) csc->colptr[i] += csc->colptr[i - 1];
  for (i = 0; i < coo->nz; i++) {
    csc->rows[csc->colptr[coo->cols[i] + 1]] = coo->rows[i];
    csc->vals[csc->colptr[coo->cols[i] + 1]++] = coo->vals[i];
  }
  assert(csc->colptr[csc->N] == coo->nz);
  return csc;
}

void free_CSC(CSC_Matrix *mat) {
  free(mat->rows);
  free(mat->colptr);
  free(mat->vals);
  free(mat);
}

CSC_Matrix *create_CSC(int rows, int cols, int nonzeros) {
  CSC_Matrix *mat = (CSC_Matrix *)malloc(sizeof(CSC_Matrix));
  mat->rows = (int *)calloc(nonzeros, sizeof(int));
  mat->colptr = (int *)calloc((cols + 2), sizeof(int));  // +2 intentional
  mat->vals = (double *)calloc(nonzeros, sizeof(double));
  mat->M = rows;
  mat->N = cols;
  mat->nz = nonzeros;
  if (mat->rows == NULL || mat->colptr == NULL || mat->vals == NULL) {
    if (mat->rows) free(mat->rows);
    if (mat->colptr) free(mat->colptr);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

int hsl_permute_and_scale(COO_Matrix *coo) {
  // step 1: convert COO to CS
  CSC_Matrix *csc = COO_to_CSC(coo);
  if (!csc) return -1;                                 // anything other than 0 is error
  const int matrix_type = (coo->isSymmetric ? 4 : 2);  // square=2, rect=1, sym=4
  assert(csc->M == csc->N);

  // step 2: find permutation and scaling (p. 3, sec 2.5 in HSL_MC64_C v.2.3.1 documentation)
  struct mc64_control control;
  struct mc64_info info;
  int *perm = (int *)malloc((coo->N + coo->M) * sizeof(int));
  double *scale = (double *)malloc((coo->N + coo->M) * sizeof(double));
  mc64_default_control(&control);  // set default control
  mc64_matching(5,                 // permute and scale
                matrix_type,       // matrix type
                csc->M,            // #rows
                csc->N,            // #cols
                csc->colptr,       // CSC column pointers
                csc->rows,         // CSC rows
                csc->vals,         // CSC values
                &control,          // MC64 Control
                &info,             // MC64 Info
                perm,              // permutation array
                scale              // scaling array
  );

  // step 3: update the COO matrix with the permutation and scaling
  if (info.flag == 0) {
    int row, col;
    double val;
    for (int nz_i = 0; nz_i < coo->nz; nz_i++) {
      row = coo->rows[nz_i];
      col = coo->cols[nz_i];
      val = coo->vals[nz_i];
      coo->rows[nz_i] = perm[row];
      coo->cols[nz_i] = perm[coo->M + col];
      coo->vals[nz_i] = exp(scale[row]) * exp(scale[coo->M + col]) * val;
    }
  } else {
    fprintf(stderr, "error HSL MC64 Permute and Scale failed with code %d\n", info.flag);
  }

  free(perm);
  free(scale);
  free_CSC(csc);
  return info.flag;
}

void count_problematic_values(COO_Matrix *mat, int *zindiag, int *zoffdiag, int *diags) {
  int in = 0, off = 0, d = 0;
  for (int i = 0; i < mat->nz; ++i) {
    if (mat->rows[i] == mat->cols[i]) {
      d++;
      if (mat->vals[i] == 0) in++;
    } else {
      if (mat->vals[i] == 0) off++;
    }
  }
  *zindiag = in;
  *zoffdiag = off;
  *diags = d;
}

int count_empty_rows(CSR_Matrix<double> *mat) {
  int ans = 0;
  for (int i = 0; i < mat->M; i++)
    if (mat->rowptr[i] == mat->rowptr[i + 1]) ans++;
  return ans;
}

void find_row_densities(CSR_Matrix<double> *mat, int *ansmin, int *ansmax, double *ansavg) {
  int min, max, cnt;
  min = max = mat->rowptr[1] - mat->rowptr[0];
  for (int i = 1; i < mat->M; i++) {
    cnt = mat->rowptr[i + 1] - mat->rowptr[i];
    if (cnt < min) min = cnt;
    if (cnt > max) max = cnt;
  }
  *ansmin = min;
  *ansmax = max;
  *ansavg = (double)mat->nz / (double)mat->M;
}

double *read_array(const char *path, const int expected_size) {
  MM_typecode matcode;
  FILE *f;
  char line[MM_MAX_LINE_LENGTH];
  int i;

  if ((f = fopen(path, "r")) == NULL) {
    return NULL;
  }
  if (mm_read_banner(f, &matcode) != 0) {
    return NULL;
  }

  // allow only certain matrices
  if (!((mm_is_real(matcode) || mm_is_integer(matcode))  // has to be real/int/binary
        && mm_is_dense(matcode)                          // has to be an array
        )) {
    return NULL;
  }

  // continue scanning until you reach the end of comments
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) return NULL;
  } while (line[0] == '%');

  // read size parameter
  int size = 0;
  sscanf(line, "%d", &size);  // not fscanf !
  assert(size == expected_size);

  // Read the values
  double *ans = (double *)malloc(sizeof(double) * size);
  for (i = 0; i < size; i++) {
    fscanf(f, "%lg\n", &ans[i]);
  }

  if (f != stdin) fclose(f);
  return ans;
}
