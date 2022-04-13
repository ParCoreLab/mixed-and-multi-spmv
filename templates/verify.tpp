#include "verify.hpp"

// Relative L2-Norm  || x - x' ||_2 / || x ||_2
template <typename T>
double L2Norm(const T *pred, const double *truth, int n) {
  double sum_diff = 0, diff, sum_truth = 0;
  for (int i = 0; i < n; ++i) {
    // || x - x' ||_2
    diff = double(pred[i]) - truth[i];
    sum_diff += diff * diff;
    // || x ||_2
    sum_truth += truth[i] * truth[i];
  }
  if (sum_truth == sum_diff && sum_truth == 0) return 0;
  return sqrt(sum_diff) / sqrt(sum_truth);
}

// Relative L1-Norm || x - x' ||_1 / || x ||_1
template <typename T>
double L1Norm(const T *pred, const double *truth, int n) {
  double sum_diff = 0, sum_truth = 0;
  for (int i = 0; i < n; ++i) {
    // || x - x' ||_1
    sum_diff += fabs(double(pred[i]) - truth[i]);
    // || x ||_1
    sum_truth += fabs(truth[i]);
  }
  if (sum_truth == sum_diff && sum_truth == 0) return 0;
  return sum_diff / sum_truth;
}

// Relative Max Norm || x - x' ||_infty / || x ||_infty
template <typename T>
double MaxNorm(const T *pred, const double *truth, int n) {
  double max_diff = 0, diff, abs_truth, max_truth = 0;
  for (int i = 0; i < n; ++i) {
    // || x - x' ||_infty
    diff = fabs(double(pred[i]) - truth[i]);
    if (diff > max_diff) max_diff = diff;
    // || x ||_infty
    abs_truth = fabs(truth[i]);
    if (abs_truth > max_truth) max_truth = abs_truth;
  }
  if (max_truth == max_diff && max_truth == 0) return 0;
  return max_diff / max_truth;
}