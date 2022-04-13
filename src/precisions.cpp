#include "precisions.hpp"

// Uses the Data-driven range based method (from the paper https://dl.acm.org/doi/abs/10.1145/3371275)
void precisions_set_datadriven(precision_e *precisions, CSR_Matrix<double> *mat, const double r, const double p,
                               split_type_e s) {
  assert(p >= 0 && p <= 100 && r >= 0);

  if (s == ENTRYWISE) {
    for (int nz_i = 0; nz_i < mat->nz; nz_i++) {
      if (mat->vals[nz_i] < r && mat->vals[nz_i] > -r) {
        precisions[nz_i] = SINGLE;
      } else {
        precisions[nz_i] = DOUBLE;
      }
    }
  } else if (s == ROWWISE) {
    for (int row = 0; row < mat->M; row++) {
      int inRange = 0, outRange = 0;
      if (mat->rowptr[row] == mat->rowptr[row + 1]) {
        // empty row can cause div by 0 below
        precisions[row] = EMPTY;
      } else {
        // calculate percentage of values in range
        bool hasPostFloatValue = false;
        for (int row_i = mat->rowptr[row]; row_i < mat->rowptr[row + 1]; row_i++) {
          // check if value is outside float range
          if (mat->vals[row_i] > FLT_MAX || mat->vals[row_i] < -FLT_MAX) {
            hasPostFloatValue = true;
            break;
          }
          // check if value is in datadriven range
          if ((mat->vals[row_i] < r && mat->vals[row_i] > -r)) {
            inRange++;
          } else {
            outRange++;
          }
        }
        if (hasPostFloatValue)
          precisions[row] = DOUBLE;  // this row has a very large value more than FLOAT_MAX, keep it in double
        else {
          const double rowperc = 100.0 * (double(inRange) / double(inRange + outRange));
          if (p == 0.0)
            precisions[row] = SINGLE;  // 0% -> choose single
          else if (p == 100.0)
            precisions[row] = (outRange == 0) ? SINGLE : DOUBLE;  // 100% -> choose single only if all are in range
          else if (inRange + outRange <= 100)
            precisions[row] = (outRange <= (100 - p)) ? SINGLE : DOUBLE;  // allow 100-p exceptions
          else
            precisions[row] = (rowperc >= p) ? SINGLE : DOUBLE;  // allow p% exceptions
        }
      }
    }
  }
}

void precisions_set_datadriven(precision_e *precisions, ELLR_Matrix<double> *mat, const double r, const double p,
                               split_type_e s) {
  assert(p >= 0 && p <= 100 && r >= 0);

  if (s == ENTRYWISE) {
    int nz_i = 0;
    for (int row = 0; row < mat->M; row++) {
      for (int row_i = mat->R * row; row_i < mat->R * row + mat->rowlen[row]; row_i++) {
        if (mat->vals[row_i] < r && mat->vals[row_i] > -r) {
          precisions[nz_i++] = SINGLE;
        } else {
          precisions[nz_i++] = DOUBLE;
        }
      }
    }

  } else if (s == ROWWISE) {
    for (int row = 0; row < mat->M; row++) {
      int inRange = 0, outRange = 0;
      if (mat->rowlen[row] == 0) {
        // empty row can cause div by 0 below
        precisions[row] = EMPTY;
      } else {
        // calculate percentage of values in range
        bool hasPostFloatValue = false;
        for (int row_i = mat->R * row; row_i < mat->R * row + mat->rowlen[row]; row_i++) {
          // check if value is outside float range
          if (mat->vals[row_i] > FLT_MAX || mat->vals[row_i] < -FLT_MAX) {
            hasPostFloatValue = true;
            break;
          }
          // check if value is in datadriven range
          if ((mat->vals[row_i] < r && mat->vals[row_i] > -r)) {
            inRange++;
          } else {
            outRange++;
          }
        }
        if (hasPostFloatValue)
          precisions[row] = DOUBLE;  // this row has a very large value more than FLOAT_MAX, keep it in double
        else {
          const double rowperc = 100.0 * (double(inRange) / double(inRange + outRange));
          if (p == 0.0)
            precisions[row] = SINGLE;  // 0% -> choose single
          else if (p == 100.0)
            precisions[row] = (outRange == 0) ? SINGLE : DOUBLE;  // 100% -> choose single only if all are in range
          else if (inRange + outRange <= 100)
            precisions[row] = (outRange <= (100 - p)) ? SINGLE : DOUBLE;  // allow 100-p exceptions
          else
            precisions[row] = (rowperc >= p) ? SINGLE : DOUBLE;  // allow p% exceptions
        }
      }
    }
  }
}

void precisions_set_custom(precision_e *precisions, CSR_Matrix<double> *mat, const int a, const int b, split_type_e s,
                           bool isContiguous) {
  assert(a <= b);
  if (isContiguous) {
    const int sep = int((double(a) / double(b)) * double(mat->M));
    // printf("SEP: %d / %d\n", sep, mat->M);
    int row = 0;
    // first block: FP32
    for (; row <= sep; ++row) {
      if (s == ROWWISE)
        precisions[row] = SINGLE;
      else  // entrywise
        for (int row_i = mat->rowptr[row]; row_i < mat->rowptr[row + 1]; row_i++) {
          precisions[row_i] = SINGLE;
        }
    }

    // second block: FP64
    for (; row < mat->M; ++row) {
      if (s == ROWWISE)
        precisions[row] = DOUBLE;
      else  // entrywise
        for (int row_i = mat->rowptr[row]; row_i < mat->rowptr[row + 1]; row_i++) {
          precisions[row_i] = DOUBLE;
        }
    }

  } else {
    int cur = 1;
    if (s == ROWWISE) {
      for (int row = 0; row < mat->M; row++) {
        if (cur <= a)
          precisions[row] = SINGLE;
        else
          precisions[row] = DOUBLE;
        if (cur < b)
          cur++;
        else
          cur = 1;
      }
    } else if (s == ENTRYWISE) {
      precision_e tmp;
      for (int row = 0; row < mat->M; row++) {
        if (cur <= a)
          tmp = SINGLE;
        else
          tmp = DOUBLE;
        if (cur < b)
          cur++;
        else
          cur = 1;
        for (int row_i = mat->rowptr[row]; row_i < mat->rowptr[row + 1]; row_i++) {
          precisions[row_i] = tmp;
        }
      }
    }
  }
}

double calculate_range(param_t params, CSR_Matrix<double> *mat) {
  // find the range for splitting
  if (params.run_option == 0) {
    // Option 0: Use constant range
    if (params.do_jacobi) {
      // If Jacobi, then use the smaller range of HSL
      return params.split_range_hsl;
    };  // dont do anything
  } else if (params.run_option == 1) {
    // Option 1: (No Scaling)  + (Absolute Mean / Factor)
    return precisions_find_absmean_range(mat, params.split_shrink_factor);
  } else
    assert(false && "error Unknown option.\n");

  return params.split_range;  // default
}

double precisions_find_absmean_range(CSR_Matrix<double> *mat, double shrink_factor) {
  // find sum of abs
  double sum = 0, curval;
  for (int i = 0; i < mat->nz; i++) {
    curval = fabs(mat->vals[i]);
    sum += curval;
  }
  sum /= (double)mat->nz;
  return sum * shrink_factor;
}