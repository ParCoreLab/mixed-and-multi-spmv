#!/bin/bash

cd "${0%/*}"
python3 --version

#Usage: architect.py 
#   <rows> <cols> 
#   <min_nzprow> <max_nzprow> 
#   <min_value> <max_value> 
#   <type: (real) r | (int) i | (binary) b> 
#   <symmetric: s | -> 
#   <diagdom: d | ->  

#                               ROWS    COLS    MIN   MAX   MIN   MAX     Type Symmetric Diagdom
# 1.476.960
python3 ../../scripts/architect.py 5000    5000    100   200   -2    2       r    s         d 
# 1.313.598
python3 ../../scripts/architect.py 32000   32000   15    25    -1    2       r    s         d 
# 3.600.686
python3 ../../scripts/architect.py 100000  100000  15    20    -1.76 1.76    r    s         d
# 2.676.854
python3 ../../scripts/architect.py 4000    4000    300   400   -2    2       r    s         d
# 154652
python3 ../../scripts/architect.py 5000    5000    10   20     -20   20      r    s         d 

