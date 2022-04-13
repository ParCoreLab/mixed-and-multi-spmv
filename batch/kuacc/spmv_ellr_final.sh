#!/bin/bash
#SBATCH --partition=ai # You can use mid / short / long if needed
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --job-name=sfe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:tesla_v100:1 # GPU resources
#SBATCH --time=06:00:00
#SBATCH --output=sfe-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etezcan19@ku.edu.tr
#SBATCH --mem=9000 # megbytes

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

echo "--- RUNNING ---"
python ./scripts/evaluator.py -a ./res/index/mm.spmvfinal.index -f spmv-ellr-final-standalone-kuacc.json \
  -c --spmv-ellr --spmv-i 2000 --split-p 99 --split-opt 1 --split-s 0.1
echo ""

echo "--- DONE ---"
date
 
