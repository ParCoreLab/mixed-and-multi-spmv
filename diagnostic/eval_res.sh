#!/bin/bash
set -xe
 
cd "${0%/*}"/..

# evaluate
python3 ./scripts/evaluator.py \
  -d ./res/ \
  -c2 -e 1e-14