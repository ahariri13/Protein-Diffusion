# Install required packages.
import os
import torch
import argparse
import json

from HNO import HNO
import datetime
today_date=datetime.datetime.now().strftime('March%d_%H-%M')
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')

# Load JSON config
with open('./config_2d.json', 'r') as f:
    config = json.load(f)


from inference_2d import inference_with_pooled

dataRef = torch.load('./dataRef.pt')

#### Checkpoint name and dataref
checkpoint_name="./model_March10_17-44.ckpt"
dataRef=dataRef.to(device)

# 2) Load the full checkpoint into this model
checkpoint = torch.load(checkpoint_name, map_location=device)

pooled_sidechain_atoms = torch.load('./Gens_diffusion/mlpgenerated_backboneSSeq_March11_2.pt').to(device)
pooled_backbone_atoms = torch.load('./Gens_diffusion//mlpgenerated_sidechainSSeq_March11_2.pt').to(device)


hidden=checkpoint['hidden']
K=checkpoint['K']
window_size=checkpoint['window_size']
num_layers=checkpoint['num_layers']

# 1) Create the same model class/architecture
model_inference = HNO(hidden,K,window_size,num_layers).to(device)

# 2) Load the full checkpoint into this model
# checkpoint = torch.load(checkpoint_name, map_location=device)
model_inference.load_state_dict(checkpoint['state_dict'])
model_inference.eval()  # set to eval mode


result = inference_with_pooled(
    model_inference,
    dataRef,
    pooled_sidechain_atoms=pooled_sidechain_atoms,
    pooled_backbone_atoms=pooled_backbone_atoms,
    device=device
)

### Save results as npy 
import numpy as np
np.save('generated_result_March11.npy',result.cpu().numpy())
# torch.save(result, './Allostery/Allos2025/Train/generated_result3.pt')


"""
Diffusion generated samples
"""

### Choose numpy prediction file 
from npy_to_pdb_2025 import read_and_reshape_npy, load_pdb, generate_pdb_files
import os
import datetime
npy_file ='./generated_result_March11.npy'


# Generate a unique folder name based on the current date and time
run_folder = f"run_{today_date}" # e.g., run_2021-02-01_12-30-45

# Create the directory
os.makedirs('./Recs_Diffusion/PDB/'+str(run_folder), exist_ok=True)

# Paths to reference PDB and base directory
pdb_file  = f"../heavy_chain.pdb"  # Adjust path
output_directory = './Recs_Diffusion/PDB/'+str(run_folder)

# Step 1: Read and reshape .npy
reshaped_array = read_and_reshape_npy(npy_file)

# Step 2: Load PDB file
all_lines, atom_lines = load_pdb(pdb_file)

# Step 3: Generate 10 PDB files
pdblines=generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)