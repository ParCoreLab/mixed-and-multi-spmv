#!/bin/bash
#SBATCH -p dgx2q
#SBATCH -N 1 
#SBATCH --job-name=ca
#SBATCH --gres=gpu:1 # GPU resources
#SBATCH --time=03:30:00
#SBATCH --output=ca-%j.out

# go to directory
cd ~/erhan/mcsr-spmv

# load modules
module load cuda11.2 gcc/9.3.0 python-3.7.4 

# build binary again
echo "--- BUILDING ---"
make clean
make CUDA_KERNEL_CHECK_FLAG=-DCUDA_CHECK_KERNELS=0
echo ""

echo "--- RUNNING ---"
date
python ./scripts/evaluator.py --cardiac -f cardiac-simula.json \
  -c --cardiac --cardiac-i 8000 --split-p 99 --split-opt 1 --split-s 2
echo ""

echo "--- DONE ---"
date
 
