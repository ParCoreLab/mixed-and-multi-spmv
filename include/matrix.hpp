#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

extern "C" {
#include "hsl_mc64d.h"  // for reordering to be used with Jacobi
#include "mmio.h"       // for input output
}

/**
 * @brief Compressed Sparse Rows format of a Sparse matrix.
 *
 * @tparam T type of the values
 */
template <typename T>
struct CSR_Matrix {
  int M, N, nz, *rowptr, *cols;
  T *vals;
};

/**
 * @brief ELLPACK-R format of a Sparse matrix.
 * R is the maximum number of elements in a row. As such, cols and vals are of size M * R.

 * @tparam T type of the values
 */
template <typename T>
struct ELLR_Matrix {
  int M, N, R, nz, *cols, *rowlen;
  T *vals;
};

/**
 * @brief Coordinate format for a Sparse matrix.
 */
typedef struct COO_Matrix {
  int M, N, nz, *rows, *cols;
  double *vals;
  bool isSymmetric;
  char type;  // 'r' = real, 'i' = integer, 'p' = pattern
} COO_Matrix;

/**
 * @brief Free COO matrix in host memory.
 *
 * @param mat COO matrix
 */
void free_COO(COO_Matrix *mat);

/**
 * @brief Allocate memory for a COO matrix in host memory.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @param nonzeros number of non-zero values
 * @return COO_Matrix*
 */
COO_Matrix *create_COO(int rows, int cols, int nonzeros);

/**
 * @brief Read a COO matrix at the given path. The matrix should be from MatrixMarket.
 *
 * @param path path to MatrixMarket matrix
 * @param isSymmetric this value will be updated to true if matrix is symetric
 * @return COO_Matrix*
 */
COO_Matrix *read_COO(const char *path);

/**
 * @brief Coordinate format for a Sparse matrix.
 */
typedef struct CSC_Matrix {
  int M, N, nz, *rows, *colptr;
  double *vals;
} CSC_Matrix;

/**
 * @brief Allocate memory for a CSC matrix in host memory.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @param nonzeros number of non-zero values
 * @return CSC_Matrix*
 */
CSC_Matrix *create_CSC(int rows, int cols, int nonzeros);

/**
 * @brief Free CSC matrix in host memory.
 *
 * @param mat CSC matrix
 */
void free_CSC(CSC_Matrix *mat);

/**
 * @brief Does HSL MC64 permutation and scaling (job=5) in double precision.
 * Mutates the input COO matrix.
 *
 * Uses MC69 within to convert COO to CSC for MC64.
 *
 * Uses HSL MC69 v1.4.1 and HSL MC64 v2.3.1.
 *
 * @param mat CSR matrix (double)
 * @return return code, 0 means success
 */
int hsl_permute_and_scale(COO_Matrix *mat);

/**
 * @brief Counts some attributes that may cause problems
 *
 * @param mat COO matrix
 * @param zindiag num-zeros in diagonal
 * @param zoffdiag num zeros off-diagonal
 * @param diags number of diagonals
 */
void count_problematic_values(COO_Matrix *mat, int *zindiag, int *zoffdiag, int *diags);

/**
 * @brief Duplicates off-diagonal entries in a symmetric matrix, where just one triangle
 * is given. Asserts the symmetry with isSymmetric ?= true.
 *
 * Implementation based on
 * https://github.com/cusplibrary/cusplibrary/blob/develop/cusp/io/detail/matrix_market.inl#L244
 *
 * @param mat
 */
void duplicate_off_diagonals(COO_Matrix *mat);

/**
 * @brief Counts the number of empty rows in the given matrix
 *
 * @param mat CSR matrix
 * @return number of empty rows
 */
int count_empty_rows(CSR_Matrix<double> *mat);

/**
 * @brief Convert a COO matrix to CSC. Does not free COO matrix.
 *
 * @param coo COO matrix
 * @return CSC_Matrix*
 */
CSC_Matrix *COO_to_CSC(COO_Matrix *coo);

/**
 * @brief Change a matrix from COO format to CSR format.
 *
 * @param coo COO matrix
 * @param compact defaults to false
 * @return CSR_Matrix<T>*
 */
CSR_Matrix<double> *COO_to_CSR(COO_Matrix *coo);

/**
 * @brief Extract the diagonal from a given CSR matrix.
 *
 * @todo This can be done inline, without the need of a new array.
 *
 * @param mat input matrix
 * @param diag diagonal
 * @param matnew mutated output matrix
 */
void extract_diagonal(CSR_Matrix<double> *mat, double *diag, CSR_Matrix<double> *matnew);

/**
 * @brief Retreive diagnostic information about the matrix.
 *
 * @param mat CSR matrix
 * @param ansmin minimum number of nnz in a row
 * @param ansmax maximum number of nnz in a row
 * @param ansavg average number of nnz in a row
 */
void find_row_densities(CSR_Matrix<double> *mat, int *ansmin, int *ansmax, double *ansavg);

/**
 * @brief Sets the range to average of max values in every row
 *
 * @param mat CSR matrix
 * @return result
 */
double avg_of_row_maxes_range(CSR_Matrix<double> *mat);

/**
 * @brief Reads an array in MatrixMarket format
 *
 * @param path path to the array
 * @param size expected size of the array (#columns of matrix)
 * @return array, allocated. User must free explicitly.
 */
double *read_array(const char *path, const int size);

// CSR Templates
template <typename T>
void free_CSR(CSR_Matrix<T> *mat);

template <typename T>
CSR_Matrix<T> *create_CSR(int rows, int cols, int nonzeros);

template <typename TSRC, typename TDEST>
CSR_Matrix<TDEST> *duplicate_CSR(CSR_Matrix<TSRC> *mat);

// ELLPACK-R Templates
template <typename T>
void free_ELLR(ELLR_Matrix<T> *mat);

template <typename T>
ELLR_Matrix<T> *create_ELLR(int rows, int cols, int nonzeros);

template <typename TSRC, typename TDEST>
ELLR_Matrix<TDEST> *duplicate_ELLR(ELLR_Matrix<TSRC> *mat);

// CSR to ELLR conversion
template <typename TSRC, typename TDEST>
ELLR_Matrix<TDEST> *CSR_to_ELLR(CSR_Matrix<TSRC> *csr);

#include "matrix.tpp"

#endif  // _MATRIX_H_