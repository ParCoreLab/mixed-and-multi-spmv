#ifndef _PRINTING_H_
#define _PRINTING_H_

#include <stdio.h>

#include "matrix.hpp"
#include "parameters.hpp"

#define PRNTSPC 16  // print space for the print functions
#define BOOL2BOOLSTR(i) (i) ? "False" : "True"

template <typename T>
void print_CSR(CSR_Matrix<T> *mat, const char *title, const char *fmt = "%.3e ");

template <typename T>
void print_ELLR(ELLR_Matrix<T> *mat, const char *title, const char *fmt = "%.3e ");

template <typename T>
void print_vector(T *vec, int N, const char *title, const char *fmt = "%.3e ");

/**
 * @brief Prints the contents of a COO matrix.
 *
 * @param mat COO matrix
 * @param title optional title for printing
 * @param fmt format string for the values. defaults to "%.3e "
 */
void print_COO(COO_Matrix *mat, const char *title, const char *fmt = "%.3e ");

/**
 * @brief Print the parameter values.
 *
 * @param params parameters
 */
void print_parameters(param_t params);

/**
 * @brief Print the results of an evaluation.
 *
 * @param params parameters
 * @param evals evaluation
 * @param title optional title for printing
 */
void print_evaluation(param_t params, eval_t evals, const char *title);

/**
 * @brief Print general information about the execution.
 *
 * @param params parameters
 * @param mat matrix
 */
void print_run_info(param_t params, CSR_Matrix<double> *mat);

/**
 * @brief Prints the header for visuals, changes depending on whether it is SpMV or Jacobi.
 *
 * @param params parameters
 * @param computetype SPMV or JACOBI
 */
void print_header(param_t params, compute_type_e computetype);

#include "printing.tpp"

#endif  // _PRINTING_H_