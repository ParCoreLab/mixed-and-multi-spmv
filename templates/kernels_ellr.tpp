#include "kernels_ellr.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ELLR Vector CUSP
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename TIN, typename TOUT, unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_CUSP(const unsigned int M,      // number of rows
                                             const unsigned int R,      // max. number of nnz in a row
                                             const TIN *Ax,             // value iterator 1, matrix values
                                             TIN *x,                    // value iterator 2, dense vector
                                             const int *Aj,             // column iterator
                                             const int *Arl,            // row lengths
                                             TOUT *y,                   // value iterator 3, result vector
                                             const bool isAccumulative  // (T): y+=Ax, (F): y=Ax
) {
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  __shared__ volatile TOUT sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
  __shared__ volatile int rl[VECTORS_PER_BLOCK];

  const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);      // thread index within the vector
  const int vector_id = thread_id / THREADS_PER_VECTOR;                // global vector index
  const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;            // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;               // total number of active vectors

  for (int row = vector_id; row < M; row += num_vectors) {
    if (thread_lane == 0) {
      rl[vector_lane] = Arl[row];
    }
    const int row_start = R * row;
    const int row_end = row_start + rl[vector_lane];

    // initialize local sum
    TOUT sum = 0.0;
    if (isAccumulative && (thread_lane == 0)) {
      sum = y[row];
    }

    if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
      int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
      if (jj >= row_start && jj < row_end) sum += Ax[jj] * x[Aj[jj]];
      for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += Ax[jj] * x[Aj[jj]];
    } else {
      for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += Ax[jj] * x[Aj[jj]];
    }

    // Store local sum in the shared memory
    sdata[threadIdx.x] = sum;

    // Reduce local sums to row sum
    TOUT tmp;
    if (THREADS_PER_VECTOR > 16) {
      tmp = sdata[threadIdx.x + 16];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 8) {
      tmp = sdata[threadIdx.x + 8];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 4) {
      tmp = sdata[threadIdx.x + 4];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 2) {
      tmp = sdata[threadIdx.x + 2];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 1) {
      tmp = sdata[threadIdx.x + 1];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }

    // First thread writes the result
    if (thread_lane == 0) {
      y[row] = sdata[threadIdx.x];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ELLR Vector Mixed
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Entrywise_Split(
    const unsigned int M,      // number of rows
    const unsigned int RS,     // max. number of nnz in a row
    const float *AxS,          // float value iterator 1, matrix values
    float *xS,                 // float value iterator 2, dense vector
    const int *AjS,            // floats column iterator
    const int *ArlS,           // floats row lengths
    const unsigned int RD,     // max. number of nnz in a row
    const double *AxD,         // double value iterator 1, matrix values
    const double *xD,          // double value iterator 2, dense vector
    const int *AjD,            // doubles column iterator
    const int *ArlD,           // doubles row lengths
    double *y,                 // value iterator 3, result vector
    const bool isAccumulative  // (T): y+=Ax, (F): y=Ax
) {
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  __shared__ volatile double
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile int rlS[VECTORS_PER_BLOCK];
  __shared__ volatile int rlD[VECTORS_PER_BLOCK];

  const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);      // thread index within the vector
  const int vector_id = thread_id / THREADS_PER_VECTOR;                // global vector index
  const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;            // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;               // total number of active vectors

  for (int row = vector_id; row < M; row += num_vectors) {
    if (thread_lane == 0) {
      rlS[vector_lane] = ArlS[row];
      rlD[vector_lane] = ArlD[row];
    }
    const int row_startS = RS * row;
    const int row_startD = RD * row;
    const int row_endS = row_startS + rlS[vector_lane];
    const int row_endD = row_startD + rlD[vector_lane];

    // initialize local sum
    double sum = 0.0;
    if (isAccumulative && (thread_lane == 0)) {
      sum = y[row];
    }

    // accumulate local sums
    // single precision
    if (THREADS_PER_VECTOR == 32 && row_endS - row_startS > 32) {
      // ensure aligned memory access to Aj and Ax
      int jj = row_startS - (row_startS & (THREADS_PER_VECTOR - 1)) + thread_lane;
      // accumulate local sums
      if (jj >= row_startS && jj < row_endS) sum += AxS[jj] * xS[AjS[jj]];
      // accumulate local sums
      for (jj += THREADS_PER_VECTOR; jj < row_endS; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[AjS[jj]];
    } else {
      // accumulate local sums
      for (int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[AjS[jj]];
    }

    // double precision
    if (THREADS_PER_VECTOR == 32 && row_endD - row_startD > 32) {
      // ensure aligned memory access to Aj and Ax
      int jj = row_startD - (row_startD & (THREADS_PER_VECTOR - 1)) + thread_lane;
      // accumulate local sums
      if (jj >= row_startD && jj < row_endD) sum += AxD[jj] * xD[AjD[jj]];
      // accumulate local sums
      for (jj += THREADS_PER_VECTOR; jj < row_endD; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[AjD[jj]];
    } else {
      // accumulate local sums
      for (int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[AjD[jj]];
    }

    // Store local sum in shared memory
    sdata[threadIdx.x] = sum;

    // Reduce local sums to row sum
    double tmp;
    if (THREADS_PER_VECTOR > 16) {
      tmp = sdata[threadIdx.x + 16];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 8) {
      tmp = sdata[threadIdx.x + 8];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 4) {
      tmp = sdata[threadIdx.x + 4];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 2) {
      tmp = sdata[threadIdx.x + 2];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 1) {
      tmp = sdata[threadIdx.x + 1];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }

    // First thread writes the result
    if (thread_lane == 0) {
      y[row] = sdata[threadIdx.x];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ELLR Vector Mixed Rowwise Clustered Split
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Split(
    const unsigned int M,      // number of rows
    const unsigned int RS,     // max. number of nnz in a row
    const float *AxS,          // float value iterator 1, matrix values
    float *xS,                 // float value iterator 2, dense vector
    const int *AjS,            // floats column iterator
    const int *ArlS,           // floats row lengths
    const unsigned int RD,     // max. number of nnz in a row
    const double *AxD,         // double value iterator 1, matrix values
    const double *xD,          // double value iterator 2, dense vector
    const int *AjD,            // doubles column iterator
    const int *ArlD,           // doubles row lengths
    double *y,                 // value iterator 3, result vector
    const int sep,             // separation index
    const bool isAccumulative  // (T): y+=Ax, (F): y=Ax
) {
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  __shared__ volatile double
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile int rl[VECTORS_PER_BLOCK];

  const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);      // thread index within the vector
  const int vector_id = thread_id / THREADS_PER_VECTOR;                // global vector index
  const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;            // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;               // total number of active vectors

  for (int row = vector_id; row < M; row += num_vectors) {
    // initialize local sum
    double sum = 0.0;
    if (isAccumulative && (thread_lane == 0)) {
      sum = y[row];
    }

    if (row < sep) {
      // single precision
      if (thread_lane == 0) {
        rl[vector_lane] = ArlS[row];
      }
      const int row_start = RS * row;
      const int row_end = row_start + rl[vector_lane];

      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxS[jj] * xS[AjS[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[AjS[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[AjS[jj]];
      }
    } else {
      // double precision
      if (thread_lane == 0) {
        rl[vector_lane] = ArlD[row - sep];  // offset separation index
      }
      const int row_start = RD * (row - sep);
      const int row_end = row_start + rl[vector_lane];

      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxD[jj] * xD[AjD[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[AjD[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[AjD[jj]];
      }
    }
    // Store local sum in shared memory
    sdata[threadIdx.x] = sum;

    // Reduce local sums to row sum
    double tmp;
    if (THREADS_PER_VECTOR > 16) {
      tmp = sdata[threadIdx.x + 16];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 8) {
      tmp = sdata[threadIdx.x + 8];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 4) {
      tmp = sdata[threadIdx.x + 4];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 2) {
      tmp = sdata[threadIdx.x + 2];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 1) {
      tmp = sdata[threadIdx.x + 1];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }

    // First thread writes the result
    if (thread_lane == 0) {
      y[row] = sdata[threadIdx.x];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ELLR Vector Mixed Rowwise Dual
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Dual(const unsigned int M,  // number of rows
                                                           const unsigned int R,  // max. number of nnz in a row
                                                           const float *AxS,   // float value iterator 1, matrix values
                                                           float *xS,          // float value iterator 2, dense vector
                                                           const double *AxD,  // double value iterator 1, matrix values
                                                           const double *xD,   // double value iterator 2, dense vector
                                                           const int *Aj,      // doubles column iterator
                                                           const int *Arl,     // doubles row lengths
                                                           double *y,          // value iterator 3, result vector
                                                           const bool *rUseSingle,    // row splits
                                                           const bool isAccumulative  // (T): y+=Ax, (F): y=Ax
) {
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  __shared__ volatile double
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile int rl[VECTORS_PER_BLOCK];
  __shared__ volatile bool useSingleShared[VECTORS_PER_BLOCK];

  const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);      // thread index within the vector
  const int vector_id = thread_id / THREADS_PER_VECTOR;                // global vector index
  const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;            // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;               // total number of active vectors

  for (int row = vector_id; row < M; row += num_vectors) {
    if (thread_lane == 0) {
      rl[vector_lane] = Arl[row];
      useSingleShared[vector_lane] = rUseSingle[row];
    }
    const int row_start = R * row;
    const int row_end = row_start + rl[vector_lane];

    // initialize local sum
    double sum = 0.0;
    if (isAccumulative && (thread_lane == 0)) {
      sum = y[row];
    }

    // accumulate local sums
    if (useSingleShared[vector_lane] == true) {
      // single precision
      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxS[jj] * xS[Aj[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[Aj[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[Aj[jj]];
      }
    } else {
      // double precision
      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxD[jj] * xD[Aj[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[Aj[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[Aj[jj]];
      }
    }

    // Store local sum in shared memory
    sdata[threadIdx.x] = sum;

    // Reduce
    double tmp;
    if (THREADS_PER_VECTOR > 16) {
      tmp = sdata[threadIdx.x + 16];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 8) {
      tmp = sdata[threadIdx.x + 8];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 4) {
      tmp = sdata[threadIdx.x + 4];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 2) {
      tmp = sdata[threadIdx.x + 2];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 1) {
      tmp = sdata[threadIdx.x + 1];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }

    // First thread writes the result
    if (thread_lane == 0) {
      y[row] = sdata[threadIdx.x];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ELLR Vector Mixed Rowwise Clustered
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <unsigned int THREADS_PER_VECTOR>
__global__ void kernel_spmv_ELLR_vector_Mixed_Rowwise_Composite(
    const unsigned int M,      // number of rows
    const unsigned int R,      // max. number of nnz in a row
    const float *AxS,          // float value iterator 1, matrix values
    float *xS,                 // float value iterator 2, dense vector
    const double *AxD,         // double value iterator 1, matrix values
    const double *xD,          // double value iterator 2, dense vector
    const int *Aj,             // doubles column iterator
    const int *Arl,            // doubles row lengths
    double *y,                 // value iterator 3, result vector
    const int sep,             // separation index
    const bool isAccumulative  // (T): y+=Ax, (F): y=Ax
) {
  const size_t VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  __shared__ volatile double
      sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile int rl[VECTORS_PER_BLOCK];

  const int thread_id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);      // thread index within the vector
  const int vector_id = thread_id / THREADS_PER_VECTOR;                // global vector index
  const int vector_lane = threadIdx.x / THREADS_PER_VECTOR;            // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;               // total number of active vectors

  for (int row = vector_id; row < M; row += num_vectors) {
    if (thread_lane == 0) {
      rl[vector_lane] = Arl[row];
    }
    const int row_start = R * row;
    const int row_end = row_start + rl[vector_lane];

    // initialize local sum
    double sum = 0.0;
    if (isAccumulative && (thread_lane == 0)) {
      sum = y[row];
    }

    // accumulate local sums
    if (row < sep) {
      // single precision
      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxS[jj] * xS[Aj[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[Aj[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxS[jj] * xS[Aj[jj]];
      }
    } else {
      // double precision
      if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
        // ensure aligned memory access to Aj and Ax
        int jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
        // accumulate local sums
        if (jj >= row_start && jj < row_end) sum += AxD[jj] * xD[Aj[jj]];
        // accumulate local sums
        for (jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[Aj[jj]];
      } else {
        // accumulate local sums
        for (int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR) sum += AxD[jj] * xD[Aj[jj]];
      }
    }

    // Store local sum in shared memory
    sdata[threadIdx.x] = sum;

    // Reduce
    double tmp;
    if (THREADS_PER_VECTOR > 16) {
      tmp = sdata[threadIdx.x + 16];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 8) {
      tmp = sdata[threadIdx.x + 8];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 4) {
      tmp = sdata[threadIdx.x + 4];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 2) {
      tmp = sdata[threadIdx.x + 2];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }
    if (THREADS_PER_VECTOR > 1) {
      tmp = sdata[threadIdx.x + 1];
      sum += tmp;
      sdata[threadIdx.x] = sum;
    }

    // First thread writes the result
    if (thread_lane == 0) {
      y[row] = sdata[threadIdx.x];
    }
  }
}