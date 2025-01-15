#!/bin/bash
#SBATCH --partition=GPU
#SBATCH -n 1
#SBATCH -ntasks-per-node 1
#SBATCH -N 1
#SBATCH --gpus-per-task=10

source myenv/bin/activate
stdbuf -o0 -e0 python3 train.py
