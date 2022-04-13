#!/bin/bash 
 
cd "${0%/*}"/..

# Command line args
if [[ $# -ne 2 ]] ; then
  echo "Please provide arguments:"
  echo "$0 <path to .mtx file> <path to log file>"
  exit 1
fi

# Check if given path exists
if [ ! -f "$1" ]; then
  echo "The given matrix '$1' does not exist."
  exit 1
fi

sudo $(which nvprof) --metrics all --log-file $2 ./bin/spmv -m $1 --prof --opt 1 --dd-p 99