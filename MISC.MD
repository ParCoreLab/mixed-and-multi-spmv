## Permutations

Suppose you have the following linear system.

```
a_1,1 * x_1 + a_1,2 * x_2 + a_1,3 * x_3 = y_1
a_2,1 * x_1 + a_2,2 * x_2 + a_2,3 * x_3 = y_2
a_3,1 * x_1 + a_3,2 * x_2 + a_3,3 * x_3 = y_3
```

This is shown as a matrix-vector multiplication of the form:

```
| a_1,1  a_1,2  a_1,3 |     | x_1 |     | a_1,1 * x_1 + a_1,2 * x_2 + a_1,3 * x_3 |     | y_1 |
| a_2,1  a_2,2  a_2,3 |  *  | x_2 |  =  | a_2,1 * x_1 + a_2,2 * x_2 + a_2,3 * x_3 |  =  | y_2 |
| a_3,1  a_3,2  a_3,3 |     | x_3 |     | a_3,1 * x_1 + a_3,2 * x_2 + a_3,3 * x_3 |     | y_3 |
```

Consider the following **row permutation** `[1 -> 2 , 2 -> 3 , 3 -> 1]`. You can't permute `x` here in a solver, because column vectors are static. But, you can't update `x` from `y` directly too, so you have the order back `y` during update phase.

```
| a_3,1  a_3,2  a_3,3 |     | x_1 |     | a_3,1 * x_1 + a_3,2 * x_2 + a_3,3 * x_3 |     | y_3 |
| a_1,1  a_1,2  a_1,3 |  *  | x_2 |  =  | a_1,1 * x_1 + a_1,2 * x_2 + a_1,3 * x_3 |  =  | y_1 |
| a_2,1  a_2,2  a_2,3 |     | x_3 |     | a_2,1 * x_1 + a_2,2 * x_2 + a_2,3 * x_3 |     | y_2 |
```

Now consider the same permutation but as a **symmetric permutation**. If you now permute `x`, you can update the solutions directly in Jacobi, although for SpMV it does not change the result.

```
| a_3,3  a_3,1  a_3,2 |     | x_3 |     | a_3,3 * x_3 + a_3,1 * x_1 + a_3,2 * x_2 |     | y_3 |
| a_1,3  a_1,1  a_1,2 |  *  | x_1 |  =  | a_1,3 * x_3 + a_1,1 * x_1 + a_1,2 * x_2 |  =  | y_1 |
| a_2,3  a_2,1  a_2,2 |     | x_2 |     | a_2,3 * x_3 + a_2,1 * x_1 + a_2,2 * x_2 |     | y_2 |
```

To summarize:

- Normally:
  - `y = Ax`
  - `x = (y - Ex) / d`
- Row Permuted:
  - `y' = A'x`
  - `x = ((y' - E'x) / d')'`
- Row + Column Permuted (Symmetric):
  - `y' = A'x'`
  - `x' = (y' - E'x') / d'`

### Swapping Vectors for Cardiac

In presence of `x32, x64 and y64`, we have 3 options for swapping:

1.  **Naive**: Call a `copying` kernel which reads `y64` and writes it to both `x32` and `x64`.
2.  **X64 Cast**: Remove `x32` from the kernel all together, and cast the value of `x64` to `float` at runtime.
3.  **X32 Copy**: At the end of `SpMV` kernel, as you write the result to `y64` write it to `x32` too. If you swap the pointers now as you normally do, `x32` will also have the swapped values.
4.  A hybrid of **1** and **3**, we swap `x64` and `y64` by pointers, but call a copy kernel on `x32` just before that. This turned out to be faster than **3**. This is what we use.

### V100 Specs

- Compute Capability: 7.0
- Warp Size: 32 Threads
- Max Warps / SM: 64
- Max Thread Blocks / SM: 32
- Max Thread Block Size: 1024

- SMs: 80
- TPCs: 40
- FP32 Cores / SM: 64
- FP64 Cores / SM: 32
- Tensor Cores / SM: 8

- Peak FP32 TFLOPS: 15.7
- Peak FP64 TFLOPs: 7.8
- Peak Tensor TFLOPS: 125

- L1 Cache Line Size: 128 B
- L2 Cache Line Size: 32 B
- L2 Cache Size: 6144 KB
- Shared Memory Size / SM: Configurable up to 96 KB

GPU cache lines are 128 bytes and are aligned. Try to make all memory accesses by warps touch the minimum number of cache lines.
See [here](https://forums.developer.nvidia.com/t/cache-line-size-of-l1-and-l2/24907) for more. Also check Ch. 5.2 of CUDA Handbook.

Also see NVIDIA [docs](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticscaches.htm) for caches: For memory cached in both L1 and L2, if every thread in a warp loads a 4-byte value from sparse locations which miss in L1 cache, each thread will incur one 128-byte L1 transaction and four 32-byte L2 transactions. This will cause the load instruction to reissue 32 times more than if the values would be adjacent and cache-aligned. If bandwidth between caches becomes a bottleneck, rearranging data or algorithms to access the data more uniformly can alleviate the problem.

Another [link](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsglobal.htm): A L1 cache line is 128 bytes and maps to a 128 byte aligned segment in device memory. Memory accesses that are cached in both L1 and L2 (cached loads using the generic data path) are serviced with 128-byte memory transactions whereas memory accesses that are cached in L2 only (uncached loads using the generic data path) are serviced with 32-byte memory transactions. Caching in L2 only can therefore reduce over-fetch, for example, in the case of scattered memory accesses.

Note that `128 bytes = 32 floats = 16 doubles`. If we are accessing less elements than that with a warp (i.e. for a row in CSR Vector SpMV), we might have worse performance.
