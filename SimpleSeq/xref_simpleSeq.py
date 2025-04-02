# Install required packages.
import os
import torch
import wandb
from torch_geometric.loader import DataLoader
import torch.nn as nn

import argparse
import json

# Load JSON config
with open('./config_2d.json', 'r') as f:
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

# Load the data
with open('../my_protein.json', 'r') as f:
    residues_data = json.load(f)

# Get a list of residue numbers (as strings)
residue_numbers = list(residues_data.keys())
# residue_numbers = sorted(residues_data.keys(), key=int)
print("Residue numbers:", residue_numbers)

res_num_str = '24'  # Use string keys
residue = residues_data[res_num_str]

num_samples=len(residues_data['1'][ 'heavy_atom_coords_per_frame']) ### 3001 = number of frames per residue. They correspond to the same residue for all 3001 samples
print(num_samples)
num_residues= 2191#len(residues_data) ### Instead of num_residues or 274 in the Ca case 
print(num_residues)

from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_graph
from utils import get_ith_frame_graph, protein_to_graph, kabsch_algorithm

protein_graphs=[]
for m in range (num_samples):
  atom_types, coords, is_backbone = get_ith_frame_graph(m,residue_numbers,residues_data)
  protein_graphs.append(protein_to_graph(coords,atom_types,is_backbone))

aligned_protein_graphs=[]
for i in range(num_samples):
  Xref = protein_graphs[0].x
  Q = protein_graphs[i].x
  _, Q_aligned = kabsch_algorithm(Xref, Q)
  ### Change x to aligned
  aligned_graph=Data(x=Q_aligned, edge_index=protein_graphs[i].edge_index,atom_type=protein_graphs[i].atom_type,backbone=protein_graphs[i].backbone)
  aligned_protein_graphs.append(aligned_graph)

dataRef_Tr=aligned_protein_graphs[0]
dataRef_Te=aligned_protein_graphs[1]

del protein_graphs

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(aligned_protein_graphs[2:], test_size=0.1, random_state=42)

train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

del aligned_protein_graphs

# from HNOSimpleSeq import HNOSimpleSeq
from SimpleSeq_V2 import SimpleSeqV2
import datetime
# today_date=datetime.datetime.now().strftime('March%d')
### Add March1 to the date+hour and minute
today_date=datetime.datetime.now().strftime('March%d_%H-%M')
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')

wandb.init(
project="Allostery_SimpleSeq",
name= str(today_date)+"_Server",
config=config,
)

# Optionally, log the args (if modified via command line)
wandb.config.update(args)

model=SimpleSeqV2(args.hidden,args.K,args.window_size,args.num_layers,args.mlp).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr
)

criterion = nn.MSELoss()

from torch_geometric.loader import DataLoader

trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

state='same' ## (Active to Active )
 
#### Checkpoint name and dataref
checkpoint_name="./model_"+str(today_date)+".ckpt"
dataRef=dataRef_Tr.to(device)
train=args.train

# checkpoint = torch.load(checkpoint_name, map_location=device)
# model.load_state_dict(checkpoint)
# model.eval()  # set to eval mode

train_pooled_backbone=[]
train_pooled_sidechain=[]
### Train
temp=1000000
if train:
  for epoch in range(args.epochs):
    # model.train()
    correct = 0
    totalLoss=0
    total_loss = 0
    N = 0
    for i, data in enumerate(trainloader):

      data=data.to(device)
      optimizer.zero_grad()

      classify,pooled_backbone_atoms,pooled_sidechain_atoms=model(data.x,data.edge_index,data.batch,data.backbone,dataRef)

      ### Only last epoch
      if epoch==args.epochs-1:
        train_pooled_backbone.append(pooled_backbone_atoms)
        train_pooled_sidechain.append(pooled_sidechain_atoms)

      ### Reconstruct either same state or State A -> B
      if state=='same':
        comparison=data.x
      else:
        comparison=data.y

      loss=criterion(classify,comparison)

      loss.backward()

      total_loss += loss.item() * data.num_graphs
      N += data.num_graphs

      optimizer.step()

      totalLoss+=loss

    totalLoss=totalLoss / (i+1)

    train_loss = total_loss / N

    # scheduler.step()

    val_correct=0
    total_val_loss=0
    Nval=0
    for j, valdata in enumerate(valoader):
      model.eval()
      valdata=valdata.to(device)

      val_classify,_,_=model(valdata.x,valdata.edge_index,valdata.batch,valdata.backbone,dataRef)

      if state=='same':
        valcomparison=valdata.x
      else:
        valcomparison=valdata.y

      val_loss = criterion(val_classify, valcomparison)

      total_val_loss += val_loss.item()*valdata.num_graphs
      Nval += valdata.num_graphs

    Val_loss = total_val_loss/Nval
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val_Loss: {Val_loss:.4f}')

    wandb.log({"Train Loss": train_loss})
    wandb.log({"Val Loss": Val_loss})
    wandb.log({"Epoch": epoch})

    if Val_loss<=temp:
      temp=Val_loss
      checkpoint = {
        'state_dict': model.state_dict(),
        'hidden': args.hidden,
        'K': args.K,
        'num_layers': args.num_layers,
        'window_size': args.window_size,
        'mlp': args.mlp,
      }
      torch.save(checkpoint, checkpoint_name)
    ### Save dataRef
    if epoch==0:
      torch.save(dataRef, "./dataRef.pt")
      print("Model saved")



### Save Poolings #####
train_pooled_backbone=torch.cat(train_pooled_backbone).detach().cpu().numpy()
train_pooled_sidechain=torch.cat(train_pooled_sidechain).detach().cpu().numpy()
#######################



checkpoint = torch.load(checkpoint_name, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # set to eval mode

total_test_loss=0
Ntest=0

real=[]
preds=[]
initial=[]
backbone_poolings=[]
sidechain_poolings=[]

dataRef_Te=dataRef_Te.to(device)

lossesCheb=[]
with torch.no_grad():
  for k, testdata in enumerate(testloader):
    model.eval()
    # model=model.to(device)
    testdata=testdata.to(device)

    test_classify,pooled_backbone_atoms,pooled_sidechain_atoms = model(testdata.x,testdata.edge_index,testdata.batch,testdata.backbone,dataRef_Tr)

    ## Append poolings of backbone and sidechain atoms
    backbone_poolings.append(pooled_backbone_atoms)
    sidechain_poolings.append(pooled_sidechain_atoms)

    test_loss = criterion(test_classify, testdata.x)

    lossesCheb.append(test_loss)

    total_test_loss += test_loss.item()*testdata.num_graphs
    Ntest += testdata.num_graphs

    real.append(testdata.x.detach().cpu())
    preds.append(torch.Tensor(test_classify).detach().cpu())

test_loss = total_test_loss/Ntest

real=torch.cat(real).numpy()
preds=torch.cat(preds).numpy()

backbone_poolings=torch.cat(backbone_poolings).detach().cpu().numpy()
sidechain_poolings=torch.cat(sidechain_poolings).detach().cpu().numpy()

wandb.log({"Test Loss": test_loss})

################
# Save the data
################

### Save the poolings in Pooling folder
# torch.save(backbone_poolings, './Allostery/Allos2025/Poolings/Backbone/pooled_backbone_atoms.pt')
# torch.save(sidechain_poolings, './Allostery/Allos2025/Poolings/SideChain/pooled_sidechain_atoms.pt')


### Save numpy
import numpy as np
# np.save('./Allostery/Allos2025/Poolings/Backbone/test_pooled_backbone_atoms.npy', backbone_poolings)
# np.save('./Allostery/Allos2025/Poolings/SideChain/test_pooled_sidechain_atoms.npy', sidechain_poolings)



### Save training poolings
np.save('./Poolings/Backbone/train_pooled_backbone_atoms_'+str(today_date)+'.npy', train_pooled_backbone)
np.save('./Poolings/SideChain/train_pooled_sidechain_atoms_'+str(today_date)+'.npy', train_pooled_sidechain)


params_list='_mlp'+str(args.mlp)+'_xref_knn'+str(args.knn)+'_K'+str(args.K)+'_window'+str(args.window_size)+'_hidden'+str(args.hidden)
# pred_file_dir='./Allostery/Allos2025/Reconstructions_Pooling/Pred/Trial'+params_list+'.npy'


### Numpy format 
import numpy as np
np.save('./Reconstructions_Pooling/Real/Trial'+params_list+'.npy',real.reshape(-1,num_residues,3))
np.save('./Reconstructions_Pooling/Pred/Trial'+params_list+'.npy',preds.reshape(-1,num_residues,3))


### Choose numpy prediction file 
from npy_to_pdb_2025 import read_and_reshape_npy, load_pdb, generate_pdb_files
npy_path = './Reconstructions_Pooling/Pred/Trial'+params_list+'.npy' #pred_file_dir

import os
import datetime

# Generate a unique folder name based on the current date and time
run_folder = f"run_{today_date+params_list}" # e.g., run_2021-02-01_12-30-45

# Create the directory
os.makedirs('./Reconstructions_Pooling/PDB/'+str(run_folder), exist_ok=True)

# Paths to reference PDB and base directory
pdb_file  = f"../heavy_chain.pdb"  # Adjust path
output_directory = './Reconstructions_Pooling/PDB/'+str(run_folder)

# Step 1: Read and reshape .npy
reshaped_array = read_and_reshape_npy(npy_path)

# Step 2: Load PDB file
all_lines, atom_lines = load_pdb(pdb_file)

# Step 3: Generate 10 PDB files
pdblines=generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)

"""
from inference_2d import inference_with_pooled

# 1) Create the same model class/architecture
model_inference = HNOSimpleSeq(args.hidden,args.K,args.window_size,args.num_layers).to(device)

# 2) Load the full checkpoint into this model
checkpoint = torch.load(checkpoint_name, map_location=device)
model_inference.load_state_dict(checkpoint)
model_inference.eval()  # set to eval mode
"""



### Create random tensors 
# pooled_main_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)
# pooled_sidechain_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)



# result = inference_with_pooled(
#     model_inference,
#     dataRef,
#     pooled_sidechain_atoms=pooled_sidechain_atoms,
#     pooled_backbone_atoms=pooled_backbone_atoms,
#     device=device
# )
# print(result.shape)

# """
# Diffusion generated samples
# """
# npy_file ='./Allostery/Allos2025/Train/generated_result.npy'


# # Generate a unique folder name based on the current date and time
# run_folder = f"run_{today_date+params_list}" # e.g., run_2021-02-01_12-30-45

# # Create the directory
# os.makedirs('./Allostery/Allos2025/Reconstructions_Diffusion/PDB/'+str(run_folder), exist_ok=True)

# # Paths to reference PDB and base directory
# pdb_file  = f"./Allostery/Allos2025/heavy_chain.pdb"  # Adjust path
# output_directory = './Allostery/Allos2025/Recs_Diffusion/PDB/'+str(run_folder)

# # Step 1: Read and reshape .npy
# reshaped_array = read_and_reshape_npy(npy_file)

# # Step 2: Load PDB file
# all_lines, atom_lines = load_pdb(pdb_file)

# # Step 3: Generate 10 PDB files
# pdblines=generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)
