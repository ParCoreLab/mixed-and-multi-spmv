#include "printing.hpp"

/**
 * @brief Prints the contents of a CSR matrix.
 *
 * @tparam T type of the values in matrix
 * @param mat CSR matrix
 * @param title optional title for printing
 * @param fmt format string for the values. defaults to "%.3e "
 */
template <typename T>
void print_CSR(CSR_Matrix<T> *mat, const char *title, const char *fmt) {
  printf("%s:\n", title);
  printf("\nVAL: ");
  for (int i = 0; i < mat->nz; ++i) {
    printf(fmt, mat->vals[i]);
  }
  printf("\nROWPTR: ");
  for (int i = 0; i < mat->M + 1; ++i) {
    printf("%d ", mat->rowptr[i]);
  }
  printf("\nCOL: ");
  for (int i = 0; i < mat->nz; ++i) {
    printf("%d ", mat->cols[i]);
  }
  printf("\n");
}

/**
 * @brief Prints the contents of a CSR matrix.
 *
 * @tparam T type of the values in matrix
 * @param mat CSR matrix
 * @param title optional title for printing
 * @param fmt format string for the values. defaults to "%.3e "
 */
template <typename T>
void print_ELLR(ELLR_Matrix<T> *mat, const char *title, const char *fmt) {
  printf("%s:\n", title);
  printf("\nVAL: ");
  for (int i = 0; i < mat->M * mat->R; ++i) {
    printf(fmt, mat->vals[i]);
  }
  printf("\nROWPTR: ");
  for (int i = 0; i < mat->M; ++i) {
    printf("%d ", mat->rowlen[i]);
  }
  printf("\nCOL: ");
  for (int i = 0; i < mat->M * mat->R; ++i) {
    printf("%d ", mat->cols[i]);
  }
  printf("\n");
}

/**
 * @brief Prints the contents of an array.
 *
 * @tparam T type of the vector
 * @param vec array
 * @param N size of the array
 * @param title optional title for printing
 * @param fmt optional print format string. defaults to "%.3e "
 */
template <typename T>
void print_vector(T *vec, int N, const char *title, const char *fmt) {
  printf("%s:\n", title);
  for (int i = 0; i < N; i++) {
    printf(fmt, vec[i]);
  }
  printf("\n");
}
