#!/bin/bash
#SBATCH --partition=ai # You can use mid / short / long if needed
#SBATCH -N 1
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --job-name=final   
#SBATCH --gres=gpu:tesla_v100:1  
#SBATCH --time=02:00:00 
#SBATCH --output=final-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etezcan19@ku.edu.tr

# go to directory
cd ~/mcsr-spmv

# load modules
module load cuda/11.2 gcc/9.3.0 python/3.7.4

# versions
echo "--- VERSIONS ---"
nvcc --version
gcc --version
python --version
echo ""

# build binary again
echo "--- BUILDING ---"
make clean
make CUDA_KERNEL_CHECK_FLAG=-DCUDA_CHECK_KERNELS=0
date
echo ""

echo "--- RUNNING SPMV ---"
python ./scripts/evaluator.py -a ./res/index/mm.spmvfinal.index -f spmv-final-kuacc.json \
  -c --spmv --spmv-i 2000 --split-p 99 --split-opt 1 --split-s 0.1 
echo ""

echo "--- RUNNING JACOBI ---"
python ./scripts/evaluator.py -a ./res/index/mm.jacobifinal.index -f jacobi-final-kuacc.json \
  -c --jacobi --jacobi-i 2000 --split-p 99 --split-opt 1 --split-s 0.1
echo ""

echo "--- DONE ---"
date
 
