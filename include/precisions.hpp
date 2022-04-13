#ifndef _PRECISIONS_H_
#define _PRECISIONS_H_

#include <assert.h>
#include <float.h>
#include <omp.h>
#include <stdbool.h>

#include "matrix.hpp"
#include "utils.hpp"

/**
 * @brief SINGLE (float, fp32) or DOUBLE (double, fp64), EMPTY is for empty rows, not used for entrywise.
 */
typedef enum { EMPTY = -1, DOUBLE = 0, SINGLE = 1 } precision_e;

/**
 * @brief Split per-entry or per-row.
 */
typedef enum { ENTRYWISE, ROWWISE } split_type_e;

/**
 * @brief Split by data-driven method.
 *
 * Uses singles for non-zero values in range -(range, range), double otherwise.
 *
 * If rowwise, it checks if a certain percentage of values are in range. If so, uses single for
 * the row, or double otherwise.
 *
 * @param precisions results are here
 * @param mat matrix
 * @param r range parameter
 * @param p percentage parameter (for rowwise)
 * @param s split type entry-wise or row-wise.
 */
void precisions_set_datadriven(precision_e *precisions, CSR_Matrix<double> *mat, const double r, const double p,
                               split_type_e s);
void precisions_set_datadriven(precision_e *precisions, ELLR_Matrix<double> *mat, const double r, const double p,
                               split_type_e s);

/**
 * @brief Get the precisions with singles assigned approximately a/b. Every a rows in b rows will be chosen single.
 * This is done instead of randomizing, to have the same deterministic split. Random with seeds had problems.
 *
 * Example: (a=1 b=4 -> 25% singles). Basically, every first a entries in b will be single. 1/2 is different than 2/4 in
 * this sense.
 *
 * @param precisions results are here
 * @param mat matrix
 * @param a numerator
 * @param b denominator, must be greater than a
 * @param s split type entry-wise or row-wise
 * @param isContiguous make the splits contiguous in memory (defaults to false)
 */
void precisions_set_custom(precision_e *precisions, CSR_Matrix<double> *mat, const int a, const int b, split_type_e s,
                           bool isContiguous = false);

/**
 * @brief Find the absolute mean of the values in a matrix.
 *
 * @param mat CSR matrix
 * @param shrink_factor how much to decrease the abs mean
 * @return absolute mean / shrink factor
 */
double precisions_find_absmean_range(CSR_Matrix<double> *mat, double shrink_factor = 3);

/**
 * @brief Returns the data-driven split range
 */
double calculate_range(param_t params, CSR_Matrix<double> *mat);

#endif  // _PRECISIONS_H_