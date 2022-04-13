#!/bin/bash
set -xe
 
cd "${0%/*}"/..

# evaluate
python3 ./scripts/evaluator.py \
  -d ./res/architect \
  -f evalarch.json \
  -c --jacobi-i 100 
  