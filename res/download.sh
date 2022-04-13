#!/bin/bash 
set -xe

cd "${0%/*}"
wget $1 -O - | tar -xvz -C ./ --strip-components=1 

# You can use MatrixMarket links at https://sparse.tamu.edu/