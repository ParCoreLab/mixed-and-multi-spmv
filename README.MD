# Exploring CSR-based Mixed and Multi-Precision SpMV for GPUs

This works aims to improve Sparse Matrix-Vector Multiplication by using mixed-precision (FP32 + FP64). In doing so, it permutes the matrix such that threads are more load balanced for the mixed-precision computations.

## Building

Use `make` to build the CUDA binary at `bin/spmv`. The compiler uses `--arch=sm_70` for NVIDIA V100, but you can change that to suit your own GPU with an `MYGPU_ARCH` environment variable, e.g. `export MYGPU_ARCH=sm_50`. We have used `cuda/11.2`, `python/3.7.4` and `gcc/9.3.0` to compile our program and run Python scripts. You also need to install and compile [`HSL_MC64`](https://www.hsl.rl.ac.uk/catalogue/mc64.html) static library with `gfortran`.

## File Structure

The file structure of this project is as follows:

- `batch` has shell scripts for cluster commands, such as queueing a job.
- `bin` for binary executables.
- `build` for build files.
- `diagnostic` has several scripts to check the program via Valgrind, cudamemcheck etc.
- `evaluations` is where we store the execution output. This is later read by Python scripts to make plots.
- `img` stores the output from Python files, such as plot images.
- `include` has header files.
- `logs` have log outputs, generally from the diagnostic tools.
- `res` has resources, such as MatrixMarket files.
- `scripts` has a variety of Python scripts, mostly for plotting and automated running of the code.
- `src` has the source files.
- `templates` has the source files for template functions.

## Running

The `Makefile` will create a binary called `spmv` under `bin` folder within the same directory, with object files under `build`. Run the executable with `-h` or `--help` option to see usage.

## Batches

For both `kuacc` and `simula` under `batches` we have the following:

- `final_experiment.sh` runs the final experiments, as used for the paper.
- `spmv_all.sh` runs SpMV test on all matrices (from `allpruned` index).
- `_srun_gpu.sh` asks for an interactive shell with one Tesla V100.
- `_check_queue.sh` checks the queue for my jobs.
- `_load_modules.sh` loads necessary modules. _does not work sometimes_

## Matrix Resources

Matrices are stored under `res` folder, with the following scripts:

- `download.sh <MatrixMarketURL>` downloads the matrix from the given URL. See [SuiteSparse](https://sparse.tamu.edu/).
- `download-from-md.sh <path>` downloads the matrices that appear in the provided Markdown file.
- `generate.sh` under `architect` generates a specific set of matrices using the `architect.py` script.
- `parsehtml.sh <path-to-html> <output-name>` parses an HTML from <http://yifanhu.net/GALLERY/GRAPHS/search.html> to create an index file.

## Diagnostics

The scripts below are under `diagnostics` folder:

- `eval_architect.sh` uses `evaluator.py` on matrices under `res/architect`.
- `eval_res.sh` uses `evaluator.py` on matrices under `res`.
- `cudamemcheck.sh` runs `cudamemcheck` with a matrix under `res/architect`.
- `valgrind.sh <matrix>` runs `valgrind` for the provided matrix.
- `nvprof.sh <matrix>` profiles SpMV kernels for the provided matrix.
- `run_random.sh` selects a random matrix under `res` and runs it.

## Scripts

Stored under `scripts` folder:

- `architect.py` creates random MatrixMarket matrices.
- `evaluator.py` runs the binary and parses it's outputs to create plots. Saves the resulting dictionary on file.
- `exporter.py` reads a a dictionary output by `evaluator.py` and exports `csv` files.
- `interpreter.py` reads a dictionary output by `evaluator.py` and plots stuff.
- `interpret.ipynb` a notebook to plot the results from another evaluation output.
- `analyser.py` analyse a specific matrix with Python.
- `plots.py` helper functions for plotting.
- `utility.py` utility functions.
- `prints.py` helper functions for printing.

`plottype` folder has generic plotting functions such as bar, heatmap, density etc. and `plotspecial` folder has specific plots.
