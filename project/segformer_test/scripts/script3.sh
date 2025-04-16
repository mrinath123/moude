#!/bin/bash

#SBATCH -p gpu22

#SBATCH -o /BS/DApt/work/Slurm_logs/run-%j.out
#SBATCH -e /BS/DApt/work/Slurm_logs/run-%j.err

#SBATCH -t 25:00:00

#SBATCH --gres gpu:2

#SBATCH -a 1-5%1

cd /BS/DApt/work/project/segformer_test;

#Make conda available:
eval "$(conda shell.bash hook)"

#Activate a conda environment:
conda activate /BS/DApt/work/build/conda/envs/seg2;

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29578  tools/train_lora_ft.py local_config/my_models/syn_75_cotta.py --launcher pytorch ${@:3}
