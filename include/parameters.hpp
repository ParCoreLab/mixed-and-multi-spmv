#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include <float.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum { SPMV = 0, JACOBI, CARDIAC, TEST } compute_type_e;

/**
 * @brief Evaluation of a single run of a certain algorithm.
 *
 */
typedef struct Evaluation {
  double time_taken_millsecs;  // Time taken by the algorithm (average ms)
  double gflops;               // GFLOPs
  double gbps;                 // GBPs
  int doubleCount;             // Number of double precision values
  int singleCount;             // Number of single count values
  double error;                // L2-Norm of residual (Jacobi) | L2-Norm of difference in results (SpMV)
  double delta;                // L1-Norm of last two guesses (Jacobi) | none (SpMV)
  double gamma;                // L2-Norm of FP64 solution and my solution (Jacobi) | none (SpMV)
  double percentage;           // (#FP32 vals)/ (#FP32 vals + #FP64 vals)
  int iterations;              // Number of iterations
  bool isConverged;            // Has converged? TODO: deprecated
  compute_type_e type;         // Computation type
} eval_t;

/**
 * @brief Parameters such as iteration count, thresholds etc.
 *
 */
typedef struct Parameters {
  int cardiac_iters;           // No. of Cardiac iterations
  int spmv_iters;              // No. of SpMV iterations
  int jacobi_iters;            // No. of Jacobi iterations
  int empty_row_count;         // Number of empty rows
  int run_option;              // Option for scaling and deciding range
  int min_nz_inrow;            // Maximum number of non-zeros in a row
  int max_nz_inrow;            // Minimum number of non-zeros in a row
  double avg_nz_inrow;         // Average number of non-zeros per row
  double split_range;          // (-range, range) range for the datadriven method
  double split_shrink_factor;  // Shrink factor for option 1
  double split_percentage;     // percentage of in-range nz per row for the datadriven method
  double split_range_hsl;      // (-range, range) range for the datadriven method (after HSL scaling)
  double doublecusp_time;      // Runtime for the CUSP Double implementation
  bool is_verbose;             // Print matrix and vector explicitly
  bool is_script;              // Is this run from within evaluator.py?
  bool is_square;              // Is matrix square?
  bool is_symmetric;           // Is matrix symmetric?
  bool do_fp64_jacobi_only;    // Do only the FP64 Jacobi
  bool do_jacobi;              // Do Jacobi
  bool do_spmv;                // Do SpMV (CSR)
  bool do_spmv_ellr;           // Do SpMV (ELLPACK-R)
  bool do_profiling;           // Do a profiling run
  bool do_cardiac;             // Do cardiac simulation
  bool read_vector;            // Read the dense vector from disk
  char mattype;                // Matrix Type (r: real, i: integer, p: pattern)
  char* matrix_path;           // Path to matrix
  char* vector_path;           // Path to vector
} param_t;

#endif  // _PARAMETERS_H_