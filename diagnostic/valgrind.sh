#!/bin/bash
 
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

#G_SLICE=always-malloc G_DEBUG=gc-friendly 
valgrind -v $(which ./bin/spmv) -m $1