#!/bin/bash
set -xe
 
cd "${0%/*}"/..

# Command line args
if [[ $# -ne 1 ]] ; then
  echo "Please provide a matrix:"
  echo "$0 <path to .mtx file>"
  exit 1
fi

# Check if given path exists
if [ ! -f "$1" ]; then
  echo "The given matrix '$1' does not exist."
  exit 1
fi

cuda-memcheck --log-file logs/cuda-memcheck.log --print-limit 0 --show-backtrace yes \
./bin/spmv -m $1 --dd-r-hsl 0.02