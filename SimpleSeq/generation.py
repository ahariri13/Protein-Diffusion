# Install required packages.
import os
import torch
import argparse
import json

from HNOSimpleSeq import HNOSimpleSeq
import datetime
today_date=datetime.datetime.now().strftime('Feb%d')
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')

# Load JSON config
with open('./Allostery/Allos2025/SimpleSeq/config_2d.json', 'r') as f:
    config = json.load(f)

# Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=config.get("hidden"))
parser.add_argument('--seed', type=int, default=config.get("seed"))
parser.add_argument('--batch_size', type=int, default=config.get("batch_size"))
parser.add_argument('--knn', type=int, default=config.get("knn"))
parser.add_argument('--K', type=int, default=config.get("K"))
parser.add_argument('--mlp', type=int, default=config.get("mlp"))
parser.add_argument('--window_size', type=int, default=config.get("window_size"))
parser.add_argument('--num_layers', type=int, default=config.get("num_layers"))
parser.add_argument('--lr', type=float, default=config.get("lr"))
parser.add_argument('--train', type=bool, default=config.get("train"))
parser.add_argument('--epochs', type=int, default=config.get("epochs"))
args = parser.parse_args()

from inference_2d import inference_with_pooled

dataRef = torch.load('./Allostery/Allos2025/SimpleSeq/dataRef.pt')

#### Checkpoint name and dataref
checkpoint_name="./Allostery/Allos2025/SimpleSeq/model_Feb28.ckpt"
dataRef=dataRef.to(device)

pooled_sidechain_atoms = torch.load('./Allostery/Allos2025/SimpleSeq/generated_sidechainSimpleSeq.pt').to(device)
pooled_backbone_atoms = torch.load('./Allostery/Allos2025/SimpleSeq/generated_backboneSimpleSeq.pt').to(device)

# 1) Create the same model class/architecture
model_inference = HNOSimpleSeq(args.hidden,args.K,args.window_size,args.num_layers).to(device)

# 2) Load the full checkpoint into this model
checkpoint = torch.load(checkpoint_name, map_location=device)
model_inference.load_state_dict(checkpoint)
model_inference.eval()  # set to eval mode

### Create random tensors 
# pooled_main_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)
# pooled_sidechain_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)

result = inference_with_pooled(
    model_inference,
    dataRef,
    pooled_sidechain_atoms=pooled_sidechain_atoms,
    pooled_backbone_atoms=pooled_backbone_atoms,
    device=device
)

### Save results as npy 
import numpy as np
np.save('./Allostery/Allos2025/SimpleSeq/generated_result.npy',result.cpu().numpy())
torch.save(result, './Allostery/Allos2025/SimpleSeq/generated_result.pt')
print(result)

print(result.shape)

"""
Diffusion generated samples
"""

### Choose numpy prediction file 
from npy_to_pdb_2025 import read_and_reshape_npy, load_pdb, generate_pdb_files
import os
import datetime
npy_file ='./Allostery/Allos2025/SimpleSeq/generated_result.npy'


# Generate a unique folder name based on the current date and time
run_folder = f"run_{today_date}" # e.g., run_2021-02-01_12-30-45

# Create the directory
os.makedirs('./Allostery/Allos2025/SimpleSeq/Recs_Diffusion/PDB/'+str(run_folder), exist_ok=True)

# Paths to reference PDB and base directory
pdb_file  = f"./Allostery/Allos2025/heavy_chain.pdb"  # Adjust path
output_directory = './Allostery/Allos2025/SimpleSeq/Recs_Diffusion/PDB/'+str(run_folder)

# Step 1: Read and reshape .npy
reshaped_array = read_and_reshape_npy(npy_file)

# Step 2: Load PDB file
all_lines, atom_lines = load_pdb(pdb_file)

# Step 3: Generate 10 PDB files
pdblines=generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)