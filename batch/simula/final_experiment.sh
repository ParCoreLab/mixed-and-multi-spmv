#!/bin/bash
#SBATCH -p dgx2q
#SBATCH -N 1 
#SBATCH --job-name=final  
#SBATCH --gres=gpu:1 # GPU resources
#SBATCH --time=01:30:00
#SBATCH --output=final-%j.out

# go to directory
cd ~/erhan/mcsr-spmv

# load modules
module load cuda11.2 gcc/9.3.0 python-3.7.4 

# build binary again
echo "--- BUILDING ---"
make clean
make CUDA_KERNEL_CHECK_FLAG=-DCUDA_CHECK_KERNELS=0
date
echo ""
 
echo "--- RUNNING SPMV ---"
python ./scripts/evaluator.py -a ./res/index/mm.spmvfinal.index -f spmv-final-simula.json \
  -c --spmv --spmv-i 2000 --split-p 99 --split-opt 1 --split-s 0.1
echo ""

echo "--- RUNNING JACOBI ---"
python ./scripts/evaluator.py -a ./res/index/mm.jacobifinal.index -f jacobi-final-simula.json \
  -c --jacobi --jacobi-i 2000 --split-p 99 --split-opt 1 --split-s 0.1 
echo ""

echo "--- DONE ---"
date
 
