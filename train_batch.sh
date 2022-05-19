#!/bin/sh
#SBATCH -J Weather
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gres gpu:1
#SBATCH -o log0321.out

python -u main.py

