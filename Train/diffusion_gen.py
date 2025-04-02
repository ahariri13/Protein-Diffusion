import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import json
from HNO import HNO
import datetime
today_date=datetime.datetime.now().strftime('March%d_%H-%M')
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
import os

model_date='March11_23-18'

backbone= np.load('./Poolings/Backbone/train_pooled_backbone_atoms_'+model_date+'.npy')
sidechain= np.load('./Poolings/SideChain/train_pooled_sidechain_atoms_'+model_date+'.npy')

checkpoint=torch.load('./model_'+model_date+'.ckpt', map_location=device)

backbone = torch.tensor(backbone).to(device)
sidechain = torch.tensor(sidechain).to(device)

def get_beta_schedule(beta_start, beta_end, diffusion_steps, schedule_type='linear'):
    """
    Returns a beta schedule tensor of shape (diffusion_steps,).
    For a linear schedule, betas are linearly spaced between beta_start and beta_end.
    For the cosine schedule, we use a cosine-based formulation.
    """
    if schedule_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
    elif schedule_type == 'cosine':
        s = 0.008
        steps = diffusion_steps
        t = torch.linspace(0, steps, steps + 1) / steps  # shape: (steps+1,)
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 1e-4, 0.999)
    else:
        raise ValueError("Unknown schedule_type. Choose 'linear' or 'cosine'.")
    return betas


class SimpleMLP(nn.Module):
    """
    A simple MLP-based denoiser.
    Operates on each token of shape (hidden) independently for an input of shape (B, k, hidden).
    """
    def __init__(self, hidden_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch_size, k, hidden_dim)
        x = F.relu(self.fc1(x))
        # x = nn.BatchNorm1d(x.size(1)).to(device)(x)
        x = F.relu(self.fc2(x))
        # x = nn.BatchNorm1d(x.size(1)).to(device)(x)
        x = F.relu(self.fc3(x))
        # x = nn.BatchNorm1d(x.size(1)).to(device)(x)
        return self.fc4(x)

class ConvDenoiser(nn.Module):
    """
    A Conv2D-based denoiser.
    The input of shape (batch_size, k, hidden) is reshaped to (batch_size, 1, k, hidden)
    so that 2D convolutions can be applied. The output is reshaped back to (B, k, hidden).
    """
    def __init__(self, k, hidden_dim):
        super(ConvDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch_size, k, hidden_dim)
        x = x.unsqueeze(1)  # Now (B, 1, k, hidden)
        x = F.relu(self.conv1(x))
        ### add batchnorm
        # x = nn.BatchNorm2d(x.size(1)).to(device)(x)
        x = F.relu(self.conv2(x))
        # x = nn.BatchNorm2d(x.size(1)).to(device)(x)

        x = F.relu(self.conv3(x))
        # x = nn.BatchNorm2d(x.size(1)).to(device)(x)

        x = self.conv4(x)
        # x = nn.BatchNorm2d(x.size(1)).to(device)(x)

        # x = F.relu(self.conv5(x))
        x = x.squeeze(1)    # Back to (B, k, hidden)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, k, hidden_dim, diffusion_steps, beta_start=0.0001, beta_end=0.02,
                 schedule_type='linear', denoiser_type='mlp', normalize=False,
                 data_mean=None, data_std=None):
        """
        k: number of tokens per sample.
        hidden_dim: feature dimension per token.
        diffusion_steps: total diffusion steps.
        beta_start, beta_end: noise schedule parameters.
        schedule_type: 'linear' or 'cosine'.
        denoiser_type: 'mlp' (default) or 'conv' for a Conv2D-based denoiser.
        normalize: if True, normalize the input data (using provided statistics) and un-normalize the generated samples.
        data_mean, data_std: normalization parameters; expected to be tensors of shape (1, 1, hidden_dim).
        """
        super(DiffusionModel, self).__init__()
        self.diffusion_steps = diffusion_steps
        beta_schedule = get_beta_schedule(beta_start, beta_end, diffusion_steps, schedule_type)
        self.register_buffer('beta_schedule', beta_schedule)
        alphas = 1 - beta_schedule
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))
        self.hidden_dim = hidden_dim
        self.k = k

        if denoiser_type == 'mlp':
            self.denoise_net = SimpleMLP(hidden_dim)
        elif denoiser_type == 'conv':
            self.denoise_net = ConvDenoiser(k, hidden_dim)
        else:
            raise ValueError("Unknown denoiser_type. Choose 'mlp' or 'conv'.")

        self.normalize = normalize
        if self.normalize:
            if data_mean is None or data_std is None:
                raise ValueError("Normalization enabled but data_mean or data_std not provided.")
            # We assume data_mean and data_std are already tensors of shape (1,1,hidden_dim)
            self.register_buffer('data_mean', data_mean)
            self.register_buffer('data_std', data_std)

    def forward(self, x0):
        """
        x0: Original samples of shape (B, k, hidden_dim).
        If normalization is enabled, the input data is normalized.
        A random time step is selected per sample, noise is added accordingly, and the
        MSE loss is computed between the predicted noise and the true noise.
        """
        if self.normalize:
            x0 = (x0 - self.data_mean) / self.data_std

        batch_size = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x0.device).long()
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        pred_noise = self.denoise_net(xt)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape, device):
        """
        Generates samples using the reverse diffusion process.
        shape: tuple (batch_size, k, hidden_dim).
        If normalization was applied during training, the generated samples are un-normalized.
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.diffusion_steps)):
            alpha_bar = self.alpha_bars[i]
            pred_noise = self.denoise_net(x)
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            if i > 0:
                alpha_bar_prev = self.alpha_bars[i - 1]
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * noise
            else:
                x = x0_pred

        if self.normalize:
            # Un-normalize the generated samples
            x = x * self.data_std + self.data_mean
        return x


def compute_dataset_mean_std(dataloader, device):
    """
    Computes per-feature mean and std for data in the dataloader.
    Assumes each sample has shape (k, hidden_dim).
    Returns tensors of shape (1, 1, hidden_dim) for mean and std.
    """
    sum_ = 0.0
    sum_sq = 0.0
    count = 0
    for batch in dataloader:
        batch = batch.to(device)
        B, k, hidden_dim = batch.shape
        count += B * k
        sum_ += batch.sum(dim=(0, 1))
        sum_sq += (batch ** 2).sum(dim=(0, 1))
    mean = sum_ / count
    var = sum_sq / count - mean ** 2
    std = torch.sqrt(var + 1e-8)
    return mean.view(1, 1, hidden_dim), std.view(1, 1, hidden_dim)

def train_diffusion_model(model, dataloader, optimizer, num_epochs, device, dataset_name=""):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"[{dataset_name}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    return model

# Main training loop for both backbone and side chain datasets.
# if __name__ == "__main__":
    # Parameters

batch_size = 64
k = checkpoint['window_size'] 
hidden_dim = checkpoint['hidden']
diffusion_steps = 100
beta_start = 1e-4
beta_end = 0.02
schedule_type = 'linear'  # or 'cosine'
# Choose denoiser_type: 'mlp' or 'conv'
denoiser_type = 'conv'
num_epochs = 50
lr=1e-3
normalize=False


# # --- Backbone Dataset ---
# backbone_dataset = backbone #RandomDataset(num_samples, k, hidden_dim)
# backbone_loader = DataLoader(backbone_dataset, batch_size=batch_size, shuffle=False)

# # Compute normalization stats from the actual backbone samples
# data_mean_backbone, data_std_backbone = compute_dataset_mean_std(backbone_loader, device)
# print("Backbone stats - mean shape:", data_mean_backbone.shape, "std shape:", data_std_backbone.shape)

# print("Training diffusion model on the backbone dataset...")
# model_backbone = DiffusionModel(k, hidden_dim, diffusion_steps, beta_start, beta_end,
#                                 schedule_type, denoiser_type, normalize=normalize,
#                                 data_mean=data_mean_backbone, data_std=data_std_backbone).to(device)
# optimizer_backbone = optim.Adam(model_backbone.parameters(), lr=lr)
# model_backbone = train_diffusion_model(model_backbone, backbone_loader, optimizer_backbone,
#                                         num_epochs, device, dataset_name="Backbone")
# model_backbone.eval()
# samples_backbone = model_backbone.sample((batch_size, k, hidden_dim), device)
# print("Generated backbone samples shape:", samples_backbone.shape)

# # --- Side Chain Dataset ---
# sidechain_dataset = sidechain#RandomDataset(num_samples, k, hidden_dim)
# sidechain_loader = DataLoader(sidechain_dataset, batch_size=batch_size, shuffle=False)

# # Compute normalization stats from the actual side chain samples
# data_mean_sidechain, data_std_sidechain = compute_dataset_mean_std(sidechain_loader, device)
# print("SideChain stats - mean shape:", data_mean_sidechain.shape, "std shape:", data_std_sidechain.shape)

# print("\nTraining diffusion model on the side chain dataset...")
# model_sidechain = DiffusionModel(k, hidden_dim, diffusion_steps, beta_start, beta_end,
#                                     schedule_type, denoiser_type, normalize=normalize,
#                                     data_mean=data_mean_sidechain, data_std=data_std_sidechain).to(device)
# optimizer_sidechain = optim.Adam(model_sidechain.parameters(), lr=lr)
# model_sidechain = train_diffusion_model(model_sidechain, sidechain_loader, optimizer_sidechain,
#                                         num_epochs, device, dataset_name="SideChain")
# model_sidechain.eval()
# samples_sidechain = model_sidechain.sample((batch_size, k, hidden_dim), device)
# print("Generated side chain samples shape:", samples_sidechain.shape)

# ### Save sichain and backbone as pt
# torch.save(samples_backbone, './Gens_diffusion/generated_backbone_'+model_date+'.pt')
# torch.save(samples_sidechain, './Gens_diffusion/generated_sidechain_'+model_date+'.pt')

################
# Generation
#################

from inference_2d import inference_with_pooled

dataRef = torch.load('./dataRef.pt')
#### Checkpoint name and dataref
checkpoint_name="./model_"+model_date+".ckpt"
dataRef=dataRef.to(device)
checkpoint = torch.load(checkpoint_name, map_location=device)


# gen_backbone= np.load('./Poolings/1D/generated_backbone_March11_23-18.npy')
# gen_sidechain= np.load('./Poolings/1D/generated_backbone_March11_23-18.npy')


# backbone= np.load('./Poolings/Backbone/train_pooled_backbone_atoms_'+model_date+'.npy')
# sidechain= np.load('./Poolings/SideChain/train_pooled_sidechain_atoms_'+model_date+'.npy')

# backbone = torch.tensor(backbone).to(device)
# sidechain = torch.tensor(sidechain).to(device)

aditya_backbone= np.load('./Gens_diffusion/generated_samples_exp11.npy')
aditya_sidechain= np.load('./Gens_diffusion/generated_samples_exp11_sc.npy')

aditya_backbone = torch.tensor(aditya_backbone).to(device)
aditya_sidechain = torch.tensor(aditya_sidechain).to(device)


gen_backbone= torch.load('./Poolings/1D/generated_backbone_March11_23-18_v3.pt')
gen_sidechain = torch.load('./Poolings/1D/generated_sidechain_March11_23-18.pt')


gen_backbone=gen_backbone[:8]       #torch.load('./Poolings/1D/generated_backbone_March11_23-18.pt')
gen_sidechain=sidechain[:8]         #torch.load('./Poolings/1D/generated_sidechain_March11_23-18.pt')

# samples_backbone=gen_backbone.reshape(-1,10,120)
# samples_sidechain=gen_sidechain.reshape(-1,10,120)


pooled_backbone_atoms =  aditya_backbone[:8].reshape(-1,10,120)    #samples_backbone   #torch.load('./Gens_diffusion//mlpgenerated_sidechainSSeq_March11_2.pt').to(device)
pooled_sidechain_atoms =  aditya_sidechain[:8].reshape(-1,10,120)    #samples_sidechain  #torch.load('./Gens_diffusion/mlpgenerated_backboneSSeq_March11_2.pt').to(device)

hidden=checkpoint['hidden']
K=checkpoint['K']
window_size=checkpoint['window_size']
num_layers=checkpoint['num_layers']

# model loading
model_inference = HNO(hidden,K,window_size,num_layers).to(device)
model_inference.load_state_dict(checkpoint['state_dict'])
model_inference.eval()  # set to eval mode

result = inference_with_pooled(
    model_inference,
    dataRef,
    pooled_sidechain_atoms=pooled_sidechain_atoms,
    pooled_backbone_atoms=pooled_backbone_atoms,
    device=device
)


gen_file='generated_result_March11_23_18.npy'

### Save results as npy 
np.save(gen_file,result.cpu().numpy())

"""
Diffusion generated samples
"""

### Choose numpy prediction file 
from npy_to_pdb_2025 import read_and_reshape_npy, load_pdb, generate_pdb_files

# npy_file ='./generated_result_March13.npy'


# Generate a unique folder name based on the current date and time
run_folder = f"run_{today_date}" # e.g., run_2021-02-01_12-30-45

# Create the directory
os.makedirs('./Recs_Diffusion/PDB/'+str(run_folder), exist_ok=True)

# Paths to reference PDB and base directory
pdb_file  = f"../heavy_chain.pdb"  # Adjust path
output_directory = './Recs_Diffusion/PDB/'+str(run_folder)

# Step 1: Read and reshape .npy
reshaped_array = read_and_reshape_npy(gen_file)

# Step 2: Load PDB file
all_lines, atom_lines = load_pdb(pdb_file)

# Step 3: Generate 10 PDB files
pdblines=generate_pdb_files(atom_lines, reshaped_array, output_directory, num_files=10)