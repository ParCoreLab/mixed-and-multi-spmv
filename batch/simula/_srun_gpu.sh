#!/bin/bash
srun -p dgx2q -N 1 --gres=gpu:1 --pty /bin/bash --login

# goes to g001 node