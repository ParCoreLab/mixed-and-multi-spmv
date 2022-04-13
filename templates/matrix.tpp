#include "matrix.hpp"

/**
 * @brief Free CSR matrix from the host memory.
 *
 * @tparam T type of the values in matrix
 * @param mat CSR matrix
 */
template <typename T>
void free_CSR(CSR_Matrix<T> *mat) {
  free(mat->rowptr);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}

/**
 * @brief Allocate memory for a CSR matrix in host memory.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @param nonzeros number of non-zero values
 */
template <typename T>
CSR_Matrix<T> *create_CSR(int rows, int cols, int nonzeros) {
  CSR_Matrix<T> *mat = (CSR_Matrix<T> *)malloc(sizeof(CSR_Matrix<T>));
  mat->rowptr = (int *)calloc((rows + 2), sizeof(int));
  mat->cols = (int *)calloc(nonzeros, sizeof(int));
  mat->vals = (T *)calloc(nonzeros, sizeof(T));
  mat->M = rows;
  mat->N = cols;
  mat->nz = nonzeros;
  if (mat->rowptr == NULL || mat->cols == NULL || mat->vals == NULL) {
    if (mat->rowptr) free(mat->rowptr);
    if (mat->cols) free(mat->cols);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

/**
 * @brief Make a copy of the given matrix, with type-casting.
 *
 * @tparam TSRC source type
 * @tparam TDEST destination type
 * @param mat source CSR matrix
 */
template <typename TSRC, typename TDEST>
CSR_Matrix<TDEST> *duplicate_CSR(CSR_Matrix<TSRC> *mat) {
  CSR_Matrix<TDEST> *matnew = create_CSR<TDEST>(mat->M, mat->N, mat->nz);
  if (!matnew) return NULL;

  for (int i = 0; i < mat->M; i++) {
    matnew->rowptr[i] = mat->rowptr[i];
    for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++) {
      matnew->cols[j] = mat->cols[j];
      matnew->vals[j] = (TDEST)mat->vals[j];
    }
  }
  matnew->rowptr[matnew->M] = mat->rowptr[mat->M];
  return matnew;
}

/**
 * @brief Free ELLPACK-R matrix from the host memory.
 *
 * @tparam T type of the values in matrix
 * @param mat ELLPACK-R matrix
 */
template <typename T>
void free_ELLR(ELLR_Matrix<T> *mat) {
  free(mat->rowlen);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}

/**
 * @brief Allocate memory for a ELLPACK-R matrix in host memory.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @param nonzeros number of non-zero values
 */
template <typename T>
ELLR_Matrix<T> *create_ELLR(int rows, int cols, int nonzeros, int R) {
  ELLR_Matrix<T> *mat = (ELLR_Matrix<T> *)malloc(sizeof(ELLR_Matrix<T>));
  mat->rowlen = (int *)malloc(rows * sizeof(int));
  mat->cols = (int *)malloc((unsigned long)rows * (unsigned long)R * (unsigned long)sizeof(int));
  mat->vals = (T *)malloc((unsigned long)rows * (unsigned long)R * (unsigned long)sizeof(T));
  mat->M = rows;
  mat->N = cols;
  mat->R = R;
  mat->nz = nonzeros;
  if (mat->rowlen == NULL || mat->cols == NULL || mat->vals == NULL) {
    if (mat->rowlen) free(mat->rowlen);
    if (mat->cols) free(mat->cols);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

/**
 * @brief Make a copy of the given matrix, with type-casting.
 *
 * @tparam TSRC source type
 * @tparam TDEST destination type
 * @param mat source ELLPACK-R matrix
 */
template <typename TSRC, typename TDEST>
ELLR_Matrix<TDEST> *duplicate_ELLR(ELLR_Matrix<TSRC> *mat) {
  ELLR_Matrix<TDEST> *matnew = create_ELLR<TDEST>(mat->M, mat->N, mat->nz, mat->R);
  if (!matnew) return NULL;

  for (int i = 0; i < mat->M; i++) {
    matnew->rowlen[i] = mat->rowlen[i];
    for (int j = mat->R * i; j < mat->R * (i + 1); j++) {
      matnew->cols[j] = mat->cols[j];
      matnew->vals[j] = (TDEST)mat->vals[j];
    }
  }
  return matnew;
}

template <typename TSRC, typename TDEST>
ELLR_Matrix<TDEST> *CSR_to_ELLR(CSR_Matrix<TSRC> *csr) {
  // find max number of elements in a row
  int R = 0;
  for (int i = 0; i < csr->M; i++) {
    if ((csr->rowptr[i + 1] - csr->rowptr[i]) > R) {
      R = csr->rowptr[i + 1] - csr->rowptr[i];
    }
  }

  // create
  ELLR_Matrix<TDEST> *ellr = create_ELLR<TDEST>(csr->M, csr->N, csr->nz, R);
  if (ellr == NULL) {
    printf("scrp error||could not allocate space for ELLPACK-R (%ld bytes)\n",
           (unsigned long)R * (unsigned long)csr->M * (unsigned long)sizeof(TDEST));
    return NULL;
  }

  // transfer
  int j_ellr = 0;
  for (int i = 0; i < csr->M; i++) {
    // row length
    ellr->rowlen[i] = csr->rowptr[i + 1] - csr->rowptr[i];

    // transfer non-zeros
    assert(j_ellr == R * i);
    for (int j = csr->rowptr[i]; j < csr->rowptr[i + 1]; j++) {
      ellr->cols[j_ellr] = csr->cols[j];
      ellr->vals[j_ellr] = csr->vals[j];
      j_ellr++;
    }
    // rest of the values are zero in this row
    while (j_ellr < R * (i + 1)) {
      ellr->cols[j_ellr] = -1;  // as invalid index
      ellr->vals[j_ellr] = 0;
      j_ellr++;
    }
  }

  return ellr;
}