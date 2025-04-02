#!/bin/bash
#SBATCH --job-name generation # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  400         # Runtime in minutes.
#SBATCH --mem 24000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage (titanrtx, gtx1080)
#SBATCH --chdir /nfs_home/hariri/Allostery/Allos2025/Train

# RUN BENCHMARK
eval "$(conda shell.bash hook)"
conda activate grop

python diffusion_gen_Train.py