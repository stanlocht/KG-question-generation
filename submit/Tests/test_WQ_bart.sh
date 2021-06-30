#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=testWQbart
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --mem=32000M
#SBATCH --output=testslurms/slurm_output_%A.out
#SBATCH -p gpu


module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Activate your environment

source venv/bin/activate
# Run your code
srun python -u test.py --savename WQbart --dataset WQ --batch_size 2 --gpus -1 --pre_trained bart \
--checkpoint "/home/stanloch/bart_WQ/default/version_1/checkpoints/epoch=8-step=21365.ckpt"