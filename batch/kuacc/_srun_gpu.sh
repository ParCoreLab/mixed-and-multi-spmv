#!/bin/bash
srun -A users --partition=ai --qos=ai --account=ai -n1 --gres=gpu:tesla_v100:1 --pty $SHELL

# -w ai12 (for a specific node)
# srun -N 1 -n1 -p short --qos=users --gres=gpu:1 -w ai12  --pty $SHELL
