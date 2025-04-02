#!/bin/bash
#SBATCH --job-name adif              # Name for your job
#SBATCH --ntasks 4                   # Number of (cpu) tasks
#SBATCH --time 4000                   # Runtime in minutes.
#SBATCH --mem 24000                  # Reserve x GB RAM for the job
#SBATCH --partition gpu              # Partition to submit
#SBATCH --qos staff                  # QOS
#SBATCH --gres gpu:titanrtx:1        # Reserve 1 GPU for usage (titanrtx, gtx1080)
#SBATCH --chdir /nfs_home/hariri/Allostery/Allos2025/SimpleSeq/Aditya  # Directory to run the job

# Activate your environment
eval "$(conda shell.bash hook)"
conda activate grop

EXP_IDX=1
EPOCHS=1000
EPOCHS_DIFF=1500

python chebnet.py --config param.yaml --debug > run_Cheb.log 2>&1

# Run your Python script and redirect both stdout and stderr to LOGFILE
python new_diff.py --config param.yaml --exp_idx $EXP_IDX --num_epochs_override $EPOCHS_DIFF --debug > run_newDiff.log 2>&1

python new_diff.py --config diffusion_backbone.yaml --exp_idx $EXP_IDX --num_epochs_override $EPOCHS_DIFF --debug > run_pure_bb_diffusion_output.log 2>&1
python new_diff.py --config diffusion_sidechain.yaml --exp_idx $EXP_IDX --num_epochs_override $EPOCHS_DIFF --debug > run_pure_sc_diffusion_output.log 2>&1
python chebnet_with_diffusion.py --config param.yaml --debug --use_diffusion --diffused_backbone_h5 latent_reps/diff_backbone/generated_diff_exp${EXP_IDX}.h5 --diffused_sidechain_h5 latent_reps/diff_sidechain/generated_diff_exp${EXP_IDX}.h5 --debug > run_Total_diffusion_output.log 2>&1

# HNO structure generation:
python h5_to_pdb.py --h5_file structures/hno_reconstructions.h5 --key hno_coords --pdb_file heavy_chain.pdb --output_dir hno --num_files 10

# Backbone (decoder 1) structure generation:
python h5_to_pdb.py --h5_file structures/backbone_coords.h5 --key backbone_coords --pdb_file backbone_heavy_chain.pdb --output_dir bb --num_files 10

# Full atom (decoder 2) structure generation:
python h5_to_pdb.py --h5_file structures/full_coords.h5 --key full_coords --pdb_file heavy_chain.pdb --output_dir sc --num_files 10

# Backbone diffusion (decoder 1) structure generation:
python h5_to_pdb.py --h5_file structures/backbone_coords_diff.h5 --key backbone_coords_diff --pdb_file backbone_heavy_chain.pdb --output_dir bb_diff --num_files 10

# Sidechain diffusion (decoder 2) structure generation:
python h5_to_pdb.py --h5_file structures/full_coords_diff.h5 --key full_coords_diff --pdb_file heavy_chain.pdb --output_dir sc_diff --num_files 10
