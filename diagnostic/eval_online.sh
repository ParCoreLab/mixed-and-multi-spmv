#!/bin/bash
set -xe
 
cd "${0%/*}"/..

# evaluate
python3 ./scripts/evaluator.py -a ./res/index/mm.test.index -c --jacobi-i 500