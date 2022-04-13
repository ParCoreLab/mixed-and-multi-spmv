#ifndef _UTILS_H_
#define _UTILS_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>  // for file_exists

#include <parameters.hpp>

// Used for command line arguments
#ifndef MATCH_INPUT
#define MATCH_INPUT(s) (!strcmp(argv[ac], (s)))
#endif

template <typename T>
T random_value(T max, T min);

template <typename T>
void write_vector(T *vec, int N, T val);

template <typename T>
void write_vector_random(T *vec, int N, T val_min, T val_max);

template <typename TSRC, typename TDEST>
void transfer_vector(TSRC *src, TDEST *dest, int N);

/**
 * @brief Relative error of a number.
 *
 * @param x input number
 * @return relative error
 */
double relative_error(double x);

/**
 * @brief Check if a file exists.
 *
 * @return true if file exists
 */
bool file_exists(char *);

#include "utils.tpp"

#endif  // _UTILS_H_