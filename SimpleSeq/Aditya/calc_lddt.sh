#!/bin/bash
#SBATCH --job-name lddt # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  400         # Runtime in minutes.
#SBATCH --mem 24000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage (titanrtx, gtx1080)
#SBATCH --chdir /nfs_home/hariri/Allostery/Allos2025/SimpleSeq/Aditya # Directory to run the job

# RUN BENCHMARK
eval "$(conda shell.bash hook)"
conda activate grop

python calc_lddt.py --h5file structures/full_coords.h5 --key full_coords --xref ref.npy


"""
calc_lddt_sample.py
====================
This script calculates lDDT scores between predicted structures (from an HDF5 file)
and a reference structure X_ref.npy, possibly restricting to the backbone only.
It also supports random subsampling of the models to compute mean/stdev of lDDT.

Key features:
    --max_samples (default=0) => if >0, randomly pick that many models from the HDF5 dataset
    --seed => set random seed for reproducibility

Usage Examples:

1) All-atom, using entire file:
    python calc_lddt_sample.py --h5file full_coords.h5 --key full_coords --xref X_ref.npy

2) Backbone-only, using entire file:
    python calc_lddt_sample.py --h5file full_coords.h5 --key full_coords --xref X_ref.npy \
      --backbone_only --pdb heavy_chain.pdb

3) Diffusion all-atom, sampling 1000 models:
    python calc_lddt_sample.py --h5file full_coords_diff.h5 --key full_coords_diff --xref X_ref.npy \
      --max_samples 1000 --seed 42

4) Diffusion backbone-only, sampling 1000 models:
    python calc_lddt_sample.py --h5file full_coords_diff.h5 --key full_coords_diff --xref X_ref.npy \
      --backbone_only --pdb heavy_chain.pdb --max_samples 1000 --seed 42

"""