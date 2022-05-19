#!/bin/sh
#SBATCH -J Weather
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gres gpu:1
#SBATCH -o log0321.out

nohup CUDA_VISIBLE_DEVICES=2,5,6,7 python -m torch.distributed.launch --nproc_per_node 4  main.py > May19.log 2>&1 &


nohup /root/runoob.sh > runoob.log 2>&1 &
29045