#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=KGQG-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurmouts/slurm_output_%A.out
#SBATCH -p gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=stanlochtenberg@gmail.com

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
srun python -u main.py --gpus -1 --batch_size 8  --max_epochs 300 --dataset WQ --test --savename WQ