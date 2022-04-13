#include "compute_utils.cuh"

/**
 * @brief Row clustering of a matrix via row-wise splits. A new matrix is created, and the input is not modified.
 *
 * The programmer is responsible of freeing the returned matrix.
 *
 * @tparam T type of the matrix
 * @param mat input matrix to be clustered
 * @param p rowwise precisions
 * @param matnew result matrix
 * @param sep separation index
 */
template <typename T>
void permute_matrix(CSR_Matrix<T> *mat, const precision_e *p, CSR_Matrix<T> *matnew, int *sep,
                    perm_type_matrix_t perm_type) {
  int nz_i = 0, i, j, i_cur = 0;

  // allocate new pointers
  matnew->M = mat->M;
  matnew->N = mat->N;
  matnew->nz = mat->nz;
  matnew->rowptr = (int *)malloc((mat->M + 1) * sizeof(int));
  matnew->cols = (int *)malloc(mat->nz * sizeof(int));
  matnew->vals = (T *)malloc(mat->nz * sizeof(T));
  matnew->rowptr[0] = 0;

  // process singles
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      matnew->rowptr[i_cur + 1] = matnew->rowptr[i_cur];
      for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
        matnew->cols[nz_i] = mat->cols[j];
        matnew->vals[nz_i] = mat->vals[j];
        matnew->rowptr[i_cur + 1]++;
        nz_i++;
      }
      i_cur++;
    }
  }

  *sep = i_cur;  // separation index

  // process doubles
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == DOUBLE) {
      matnew->rowptr[i_cur + 1] = matnew->rowptr[i_cur];
      for (j = mat->rowptr[i]; j < mat->rowptr[i + 1]; ++j) {
        matnew->cols[nz_i] = mat->cols[j];
        matnew->vals[nz_i] = mat->vals[j];
        matnew->rowptr[i_cur + 1]++;
        nz_i++;
      }
      i_cur++;
    }
  }

  // empty rows are ignored
  for (i = 0; i < mat->M; ++i) {
    if (p[i] == EMPTY) {
      matnew->rowptr[i_cur + 1] = matnew->rowptr[i_cur];
      i_cur++;
    }
  }

  // If symmetric, we need to permute the columns too.
  if (perm_type == PERMUTE_SYMMETRIC) {
    int *perm = (int *)malloc(mat->M * sizeof(int));
    get_permutations(p, mat->M, perm);

    // permute
    for (i = 0; i < matnew->M; ++i) {
      for (j = matnew->rowptr[i]; j < matnew->rowptr[i + 1]; ++j) {
        matnew->cols[j] = perm[matnew->cols[j]];
      }
    }

    free(perm);
  }

  // assertions
  assert(nz_i == mat->nz);
  assert(i_cur == mat->M);
  assert(matnew->rowptr[matnew->M] == mat->rowptr[mat->M]);
}

template <typename T>
void permute_matrix(ELLR_Matrix<T> *mat, const precision_e *p, ELLR_Matrix<T> *matnew, int *sep,
                    perm_type_matrix_t perm_type) {
  int j_cur = 0, i_cur = 0;

  // allocate new pointers
  matnew->M = mat->M;
  matnew->N = mat->N;
  matnew->R = mat->R;
  matnew->nz = mat->nz;
  matnew->rowlen = (int *)malloc(mat->M * sizeof(int));
  matnew->cols = (int *)malloc(mat->M * mat->R * sizeof(int));
  matnew->vals = (T *)malloc(mat->M * mat->R * sizeof(T));

  // process singles
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == SINGLE) {
      matnew->rowlen[i_cur] = mat->rowlen[i];
      for (int j = mat->R * i; j < mat->R * (i + 1); ++j) {
        matnew->cols[j_cur] = mat->cols[j];
        matnew->vals[j_cur] = mat->vals[j];
        j_cur++;
      }
      i_cur++;
    }
  }

  *sep = i_cur;  // separation index

  // process doubles
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == DOUBLE) {
      matnew->rowlen[i_cur] = mat->rowlen[i];
      for (int j = mat->R * i; j < mat->R * (i + 1); ++j) {
        matnew->cols[j_cur] = mat->cols[j];
        matnew->vals[j_cur] = mat->vals[j];
        j_cur++;
      }
      i_cur++;
    }
  }

  // process empty rows
  for (int i = 0; i < mat->M; ++i) {
    if (p[i] == EMPTY) {
      matnew->rowlen[i_cur] = mat->rowlen[i];
      assert(matnew->rowlen[i_cur] == 0);
      for (int j = mat->R * i; j < mat->R * (i + 1); ++j) {
        matnew->cols[j_cur] = mat->cols[j];
        matnew->vals[j_cur] = mat->vals[j];
        j_cur++;
      }
      i_cur++;
    }
  }

  // If symmetric, we need to permute the columns too.
  if (perm_type == PERMUTE_SYMMETRIC) {
    int *perm = (int *)malloc(mat->M * sizeof(int));
    get_permutations(p, mat->M, perm);

    // permute
    for (int i = 0; i < matnew->M; ++i) {
      for (int j = matnew->R * i; j < matnew->R * i + matnew->rowlen[i]; ++j) {
        matnew->cols[j] = perm[matnew->cols[j]];
      }
    }

    free(perm);
  }

  // assertions
  assert(j_cur == mat->R * mat->M);
  assert(i_cur == mat->M);
}

/**
 * @brief Permute a vector.
 *
 * The programmer is responsible from freeing the returned vector.
 *
 * @tparam T type of the values in the vector
 * @param vec vector pointer
 * @param len size of the vector
 * @param p row precisions
 * @param forward is this a new cluster or reversing an existing cluster? Defaults to the former.
 */
template <typename T>
void permute_vector(const T *vec, const int len, const precision_e *p, T *vecnew, perm_type_vector_t perm_type) {
  int v_i = 0, i;
  if (perm_type == PERMUTE_FORWARD) {
    // singles
    for (i = 0; i < len; i++)
      if (p[i] == SINGLE) vecnew[v_i++] = vec[i];

    // doubles
    for (i = 0; i < len; i++)
      if (p[i] == DOUBLE) vecnew[v_i++] = vec[i];

    // empties
    for (i = 0; i < len; i++)
      if (p[i] == EMPTY) vecnew[v_i++] = vec[i];

    assert(v_i == len);
  } else {
    // singles
    for (i = 0; i < len; i++)
      if (p[i] == SINGLE) vecnew[i] = vec[v_i++];

    // doubles
    for (i = 0; i < len; i++)
      if (p[i] == DOUBLE) vecnew[i] = vec[v_i++];

    // empties
    for (i = 0; i < len; i++)
      if (p[i] == EMPTY) vecnew[i] = vec[v_i++];

    assert(v_i == len);
  }
}

/**
 * @brief Sum of all the values in an array
 */
template <typename T>
T sum_vector(T *vec, int N) {
  T sum = 0;
  for (int i = 0; i < N; i++) {
    sum += vec[i];
  }
  return sum;
}