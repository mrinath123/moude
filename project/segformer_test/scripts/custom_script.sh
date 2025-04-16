#!/bin/bash

#SBATCH -p gpu22,gpu20,gpu16

#SBATCH -o /BS/DApt/work/Slurm_logs/run-%j.out
#SBATCH -e /BS/DApt/work/Slurm_logs/run-%j.err

#SBATCH -t 16:00:00

#SBATCH --gres gpu:2

#SBATCH -a 1-5%1

cd /BS/DApt/work/project/segformer_test;

#Make conda available:
eval "$(conda shell.bash hook)"

#Activate a conda environment:
conda activate /BS/DApt/work/build/conda/envs/seg2;

python3 tent.py