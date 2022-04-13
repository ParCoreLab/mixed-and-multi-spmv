#ifndef _VERIFY_H_
#define _VERIFY_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T>
double L2Norm(const T *pred, const double *truth, int n);

template <typename T>
double L1Norm(const T *pred, const double *truth, int n);

template <typename T>
double MaxNorm(const T *pred, const double *truth, int n);

#include "verify.tpp"

#endif  // _VERIFY_H_