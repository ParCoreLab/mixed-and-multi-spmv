#!/bin/bash

# come to this directory
cd "${0%/*}"

# copy
scp -r etezcan19@login.kuacc.ku.edu.tr:/kuacc/users/etezcan19/mcsr-spmv/batch/*.out ./slurms/