import os
import sys
import json
import yaml
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py  # for final outputs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph
from sklearn.model_selection import train_test_split

#################################################################
# Argument Parsing
#################################################################
parser = argparse.ArgumentParser(description="Protein Reconstruction with Pretrained HNO & Two-Step Decoders")
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

# Optional flags to override pooled embeddings with diffusion
parser.add_argument("--use_diffusion", action="store_true",
                    help="If set, attempt to override the internal pooling with diffused embeddings at final export.")
parser.add_argument("--diffused_backbone_h5", type=str, default=None,
                    help="Path to HDF5 with diffused backbone embeddings (dataset=backbone_pooled).")
parser.add_argument("--diffused_sidechain_h5", type=str, default=None,
                    help="Path to HDF5 with diffused sidechain embeddings (dataset=sidechain_pooled).")

args = parser.parse_args()

# --- Logging Setup ---
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
use_debug = config.get("use_debug_logs", False) or args.debug
log_file = config.get("log_file", "chebnetdiffusion_logging.log")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger("ProteinReconstruction")
logger.setLevel(logging.DEBUG if use_debug else logging.INFO)
fh = logging.FileHandler(log_file, mode="w")
fh.setLevel(logging.DEBUG if use_debug else logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if use_debug else logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Logger initialized.")
if use_debug:
    logger.debug("Debug mode is ON.")
else:
    logger.info("Debug mode is OFF; minimal logs will be shown.")

#################################################################
# Utility: Checkpoint Save/Load
#################################################################
def save_checkpoint(state, filename, logger):
    torch.save(state, filename)
    logger.debug(f"Checkpoint saved to {filename}")
    sys.stdout.flush()

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    if os.path.isfile(filename):
        logger.info(f"Loading checkpoint from '{filename}'")
        sys.stdout.flush()
        checkpoint = torch.load(filename, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint at epoch {start_epoch}")
        sys.stdout.flush()
    else:
        logger.info(f"No checkpoint found at '{filename}'. Training from scratch.")
        sys.stdout.flush()
    return model, optimizer, start_epoch

#################################################################
# (A) PDB Parsing & Backbone/Sidechain Extraction
#################################################################
def parse_pdb(filename):
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    original_dict = {}
    atoms_in_order = []
    with open(filename, 'r') as pdb_file:
        for line in pdb_file:
            if not line.startswith("ATOM"):
                continue
            tokens = line.split()
            if len(tokens) < 6:
                continue
            try:
                orig_atom_index = int(tokens[1])
            except ValueError:
                continue
            atom_name = tokens[2]
            orig_res_id = tokens[5]
            category = "backbone" if atom_name in backbone_atoms else "sidechain"
            if orig_res_id not in original_dict:
                original_dict[orig_res_id] = {"backbone": [], "sidechain": []}
            original_dict[orig_res_id][category].append(orig_atom_index)
            atoms_in_order.append((orig_res_id, orig_atom_index, category))
    return original_dict, atoms_in_order

def renumber_atoms_and_residues(atoms_in_order):
    new_dict = {}
    res_mapping = {}
    next_new_res_id = 0
    next_new_atom_index = 0
    for (orig_res_id, orig_atom_index, category) in atoms_in_order:
        if orig_res_id not in res_mapping:
            res_mapping[orig_res_id] = next_new_res_id
            new_dict[next_new_res_id] = {"backbone": [], "sidechain": []}
            next_new_res_id += 1
        new_res_id = res_mapping[orig_res_id]
        new_dict[new_res_id][category].append(next_new_atom_index)
        next_new_atom_index += 1
    return new_dict

def get_global_indices(renumbered_dict):
    backbone_indices, sidechain_indices = [], []
    for res_id in sorted(renumbered_dict.keys()):
        backbone_indices.extend(renumbered_dict[res_id]["backbone"])
        sidechain_indices.extend(renumbered_dict[res_id]["sidechain"])
    return sorted(backbone_indices), sorted(sidechain_indices)

#################################################################
# (B) Load JSON heavy-atom coordinates
#################################################################
def load_heavy_atom_coords_from_json(json_file, logger):
    logger.info(f"Loading JSON from {json_file}")
    with open(json_file, "r") as f:
        data = json.load(f)
    residue_keys_sorted = sorted(data.keys(), key=lambda x: int(x))
    num_frames = len(data[residue_keys_sorted[0]]["heavy_atom_coords_per_frame"])
    logger.info(f"Number of frames in JSON: {num_frames}")
    coords_per_frame = []
    for frame_idx in range(num_frames):
        frame_coords_list = []
        for res_key in residue_keys_sorted:
            coords_this_res = data[res_key]["heavy_atom_coords_per_frame"][frame_idx]
            frame_coords_list.append(np.array(coords_this_res, dtype=np.float32))
        frame_coords = np.concatenate(frame_coords_list, axis=0)
        coords_per_frame.append(torch.tensor(frame_coords, dtype=torch.float32))
    return coords_per_frame

#################################################################
# (C) Kabsch Alignment
#################################################################
def compute_centroid(X):
    return X.mean(dim=0)

def kabsch_algorithm(P, Q):
    centroid_P = compute_centroid(P)
    centroid_Q = compute_centroid(Q)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    C = torch.mm(Q_centered.T, P_centered)
    V, S, W = torch.svd(C)
    d = torch.det(torch.mm(V, W.T))
    if d < 0:
        V[:, -1] = -V[:, -1]
    U = torch.mm(V, W.T)
    Q_aligned = torch.mm(Q_centered, U) + centroid_P
    return U, Q_aligned

def align_frames_to_first(coords_list, logger):
    logger.info("Aligning all frames to the first frame via Kabsch...")
    reference = coords_list[0]
    aligned = [reference]
    logger.debug(f"Reference frame shape: {reference.shape}")
    for i, coords in enumerate(coords_list[1:], start=1):
        _, coords_aligned = kabsch_algorithm(reference, coords)
        aligned.append(coords_aligned)
        if i < 5:
            logger.debug(f"Frame {i}: aligned shape = {coords_aligned.shape}")
    return aligned

#################################################################
# (D) Build PyG Graph Dataset
#################################################################
def build_graph_dataset(coords_list, knn_neighbors=4, logger=None):
    if logger:
        logger.info("Building PyG dataset using knn_graph.")
    dataset = []
    for coords in coords_list:
        edge_index = knn_graph(coords, k=knn_neighbors, loop=False)
        data = Data(x=coords, edge_index=edge_index, y=coords)
        dataset.append(data)
    return dataset

#################################################################
# (E) Blind Pooling Module (2D)
#################################################################
class BlindPooling2D(nn.Module):
    """
    A simple 2D pooling block that takes [B, N, emb_dim] -> [B, H*W] via AdaptiveAvgPool2d.
    """
    def __init__(self, H, W):
        super().__init__()
        self.pool2d = nn.AdaptiveAvgPool2d((H, W))
        self.H = H
        self.W = W

    def forward(self, x):
        B, N, E = x.shape
        x_4d = x.unsqueeze(1)  # [B, 1, N, E]
        pooled = self.pool2d(x_4d)  # [B, 1, H, W]
        pooled_flat = pooled.view(B, self.H * self.W)
        return pooled_flat

#################################################################
# (F) HNO Model (ChebConv based)
#################################################################
class HNO(nn.Module):
    def __init__(self, hidden_dim, K):
        super().__init__()
        self._debug_logged = False  # For one-time debug logging
        logger.debug(f"Initializing HNO with hidden_dim={hidden_dim}, K={K}")
        sys.stdout.flush()
        self.conv1 = ChebConv(3, hidden_dim, K=K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv4 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bano1 = nn.BatchNorm1d(hidden_dim)
        self.bano2 = nn.BatchNorm1d(hidden_dim)
        self.bano3 = nn.BatchNorm1d(hidden_dim)
        self.mlpRep = nn.Linear(hidden_dim, 3)

    def forward(self, x, edge_index, log_debug=False):
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] Input x shape: {x.shape}")

        x = self.conv1(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv1: {x.shape}")
        x = F.leaky_relu(x)
        x = self.bano1(x)

        x = self.conv2(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv2: {x.shape}")
        x = F.leaky_relu(x)
        x = self.bano2(x)

        x = self.conv3(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv3: {x.shape}")
        x = F.relu(x)
        x = self.bano3(x)

        x = self.conv4(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] After conv4: {x.shape}")

        x = F.normalize(x, p=2, dim=1)
        x = self.mlpRep(x)

        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO] Output (after mlpRep): {x.shape}")
            self._debug_logged = True

        return x

    def forward_representation(self, x, edge_index, log_debug=False):
        """
        Returns the latent representation after the final conv (before final MLP).
        """
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] Input x shape: {x.shape}")

        x = self.conv1(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] After conv1: {x.shape}")
        x = F.leaky_relu(x)
        x = self.bano1(x)

        x = self.conv2(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] After conv2: {x.shape}")
        x = F.leaky_relu(x)
        x = self.bano2(x)

        x = self.conv3(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] After conv3: {x.shape}")
        x = F.relu(x)
        x = self.bano3(x)

        x = self.conv4(x, edge_index)
        if log_debug and not self._debug_logged:
            logger.debug(f"[HNO.rep] After conv4: {x.shape}")
            self._debug_logged = True

        x = F.normalize(x, p=2, dim=1)
        return x

#################################################################
# Train HNO
#################################################################
def train_hno_model(model, train_loader, test_loader, num_epochs, learning_rate, checkpoint_path, save_interval=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    logger.info(f"Starting HNO from epoch={start_epoch}, total epochs={num_epochs}, LR={learning_rate}")
    sys.stdout.flush()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss_val = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index)
            loss = criterion(pred, data.x)
            loss.backward()
            optimizer.step()
            train_loss_val += loss.item()
        avg_train_loss = train_loss_val / len(train_loader)

        model.eval()
        test_loss_val = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index)
                loss = criterion(pred, data.x)
                test_loss_val += loss.item()
        avg_test_loss = test_loss_val / len(test_loader)

        logger.info(f"[HNO] Epoch {epoch+1}/{num_epochs} => TRAIN MSE={avg_train_loss:.6f}, TEST MSE={avg_test_loss:.6f}")
        sys.stdout.flush()

        if (epoch + 1) % save_interval == 0:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path, logger)
            logger.info(f"HNO checkpoint saved at epoch {epoch+1} -> {checkpoint_path}")
            sys.stdout.flush()
    return model

#################################################################
# Helper: Build simple MLP
#################################################################
def build_mlp(input_dim, output_dim, hidden_dim=128, num_layers=2, use_layernorm=True):
    layers = []
    in_dim = input_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

#################################################################
# (G) Backbone Decoder
#     - Optional override: override_pooled_backbone
#################################################################
class BackboneDecoder(nn.Module):
    def __init__(
        self,
        num_total_atoms,
        backbone_indices,
        emb_dim,
        pooling_dim=(20,4),
        mlp_depth=2
    ):
        super().__init__()
        self._debug_logged = False
        self.num_total_atoms = num_total_atoms
        self.backbone_indices = backbone_indices
        self.backbone_count = len(backbone_indices)

        self.pool_backbone = BlindPooling2D(*pooling_dim)
        self.pool_dim = pooling_dim[0] * pooling_dim[1]

        self.input_dim = self.backbone_count * (emb_dim + self.pool_dim)
        self.output_dim = self.backbone_count * 3

        self.mlp_flat = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=128,
            num_layers=mlp_depth,
            use_layernorm=True
        )

    def forward(self, hno_latent, z_ref, log_debug=False, override_pooled_backbone=None):
        """
        If override_pooled_backbone is given => skip internal pooling, use that tensor [B, pool_dim].
        """
        B_times_N, E = hno_latent.shape
        B = B_times_N // self.num_total_atoms
        should_log = (log_debug and not self._debug_logged)

        x = hno_latent.view(B, self.num_total_atoms, E)
        backbone_emb = x[:, self.backbone_indices, :]

        if override_pooled_backbone is None:
            pooled_backbone = self.pool_backbone(backbone_emb)  # normal pooling
        else:
            # dimension check
            if override_pooled_backbone.shape != (B, self.pool_dim):
                raise ValueError(f"override_pooled_backbone has shape {override_pooled_backbone.shape}, expected ({B}, {self.pool_dim})")
            pooled_backbone = override_pooled_backbone

        if z_ref is None:
            raise ValueError("[BackboneDecoder] z_ref is None. Must pass a valid tensor.")
        z_ref_expanded = z_ref.unsqueeze(0).expand(B, -1, -1)
        z_ref_backbone = z_ref_expanded[:, self.backbone_indices, :]

        combined = torch.cat([
            z_ref_backbone, 
            pooled_backbone.unsqueeze(1).expand(-1, self.backbone_count, -1)
        ], dim=-1)
        combined_flat = combined.view(B, self.backbone_count*(E + self.pool_dim))
        pred_bb_flat = self.mlp_flat(combined_flat)
        pred_bb = pred_bb_flat.view(B, self.backbone_count, 3)

        if should_log:
            logger.debug(f"[BackboneDecoder] pred_bb => {pred_bb.shape}")
            self._debug_logged = True

        return pred_bb

#################################################################
# (H) Sidechain Decoder
#     - Optional override: override_pooled_sidechain
#################################################################
class SidechainDecoder(nn.Module):
    def __init__(
        self,
        num_total_atoms,
        sidechain_indices,
        backbone_indices,
        emb_dim,
        pooling_dim=(20,4),
        mlp_depth=2,
        arch_type=0
    ):
        super().__init__()
        self._debug_logged = False
        self.num_total_atoms = num_total_atoms
        self.sidechain_indices = sidechain_indices
        self.backbone_indices = backbone_indices
        self.sidechain_count = len(sidechain_indices)
        self.backbone_count = len(backbone_indices)
        self.arch_type = arch_type

        self.pool_sidechain = BlindPooling2D(*pooling_dim)
        self.pool_dim = pooling_dim[0] * pooling_dim[1]

        if arch_type >= 1:
            self.sc_zref_reduce = nn.Linear(self.sidechain_count * emb_dim, 128)
        else:
            self.sc_zref_reduce = None

        if arch_type == 2:
            self.bb_reduce = nn.Linear(self.backbone_count * 3, 128)
        else:
            self.bb_reduce = None

        # final_in_dim logic
        base_in_dim = self.backbone_count * 3 + self.pool_dim
        extra_in_dim = 0
        if arch_type >= 1:
            extra_in_dim += 128
        if arch_type == 2:
            base_in_dim = self.pool_dim
            extra_in_dim += 128
        final_in_dim = base_in_dim + extra_in_dim

        self.mlp_sidechain = build_mlp(
            input_dim=final_in_dim,
            output_dim=self.sidechain_count * 3,
            hidden_dim=128,
            num_layers=mlp_depth,
            use_layernorm=True
        )

    def forward(self, hno_latent, predicted_backbone, z_ref, log_debug=False,
                override_pooled_sidechain=None):
        B_times_N, E = hno_latent.shape
        B = B_times_N // self.num_total_atoms
        should_log = (log_debug and not self._debug_logged)

        x = hno_latent.view(B, self.num_total_atoms, E)
        sidechain_emb = x[:, self.sidechain_indices, :]

        if override_pooled_sidechain is None:
            pooled_sidechain = self.pool_sidechain(sidechain_emb)
        else:
            if override_pooled_sidechain.shape != (B, self.pool_dim):
                raise ValueError(f"override_pooled_sidechain has shape {override_pooled_sidechain.shape}, expected ({B}, {self.pool_dim})")
            pooled_sidechain = override_pooled_sidechain

        bb_flat = predicted_backbone.view(B, self.backbone_count * 3)
        bb_reduced = None
        if self.arch_type == 2 and self.bb_reduce is not None:
            bb_reduced = self.bb_reduce(bb_flat)

        sc_zref_reduced = None
        if self.arch_type >= 1:
            if z_ref is None:
                raise ValueError("[SidechainDecoder] arch_type>=1 but z_ref is None!")
            z_ref_batch = z_ref.unsqueeze(0).expand(B, -1, -1)
            sc_zref = z_ref_batch[:, self.sidechain_indices, :]
            sc_zref_flat = sc_zref.view(B, self.sidechain_count * E)
            sc_zref_reduced = self.sc_zref_reduce(sc_zref_flat)

        if self.arch_type == 0:
            final_input = torch.cat([bb_flat, pooled_sidechain], dim=-1)
        elif self.arch_type == 1:
            final_input = torch.cat([bb_flat, pooled_sidechain, sc_zref_reduced], dim=-1)
        else:
            final_input = torch.cat([bb_reduced, pooled_sidechain, sc_zref_reduced], dim=-1)

        sidechain_coords_flat = self.mlp_sidechain(final_input)
        pred_sidechain_coords = sidechain_coords_flat.view(B, self.sidechain_count, 3)

        full_coords = torch.zeros(B, self.num_total_atoms, 3, device=sidechain_coords_flat.device)
        full_coords[:, self.backbone_indices, :] = predicted_backbone
        full_coords[:, self.sidechain_indices, :] = pred_sidechain_coords

        if should_log:
            logger.debug(f"[SidechainDecoder] final full_coords => {full_coords.shape}")
            self._debug_logged = True

        return full_coords

#################################################################
# (I) Training routines
#################################################################
def train_backbone_decoder(model, train_loader, test_loader, device, logger, config, checkpoint_path, z_ref):
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    epochs = config["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    backbone_indices = model.backbone_indices
    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            log_flag = (i == 0)

            pred_bb = model(data.x, z_ref=z_ref.to(device), log_debug=log_flag)
            B_times_N, _ = data.y.shape
            B = B_times_N // model.num_total_atoms
            coords_3d = data.y.view(B, model.num_total_atoms, 3)
            gt_backbone = coords_3d[:, backbone_indices, :]

            loss = criterion(pred_bb, gt_backbone)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Test
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred_bb = model(data.x, z_ref=z_ref.to(device), log_debug=False)
                B_times_N, _ = data.y.shape
                B = B_times_N // model.num_total_atoms
                coords_3d = data.y.view(B, model.num_total_atoms, 3)
                gt_backbone = coords_3d[:, backbone_indices, :]
                val_loss = criterion(pred_bb, gt_backbone)
                total_test_loss += val_loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        if (epoch+1) % config.get("log_interval", 10) == 0 or (epoch+1)==epochs:
            logger.info(f"[BackboneDecoder] Epoch {epoch+1}/{epochs} => TRAIN_BB_MSE: {avg_train_loss:.4f}, TEST_BB_MSE: {avg_test_loss:.4f}")

        if (epoch+1) % config.get("save_interval", 10) == 0:
            checkpoint_state = {
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path, logger)

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    logger.info(f"Backbone decoder checkpoint saved to {checkpoint_path}")
    return model

def train_sidechain_decoder(model, train_loader, test_loader, backbone_decoder, device, logger, config, checkpoint_path, z_ref):
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    epochs = config["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_atoms = model.num_total_atoms
    backbone_indices = model.backbone_indices
    sidechain_indices = model.sidechain_indices

    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    model = model.to(device)
    backbone_decoder = backbone_decoder.to(device)
    backbone_decoder.eval()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        total_train_bb_mse = 0.0
        total_train_sc_mse = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            log_flag = (i == 0 and epoch==start_epoch)

            with torch.no_grad():
                pred_bb = backbone_decoder(data.x, z_ref=z_ref.to(device), log_debug=log_flag)
            full_pred = model(data.x, pred_bb, z_ref=z_ref.to(device), log_debug=log_flag)

            B_times_N, _ = data.y.shape
            B = B_times_N // num_atoms
            coords_3d = data.y.view(B, num_atoms, 3)
            bb_pred = full_pred[:, backbone_indices, :]
            sc_pred = full_pred[:, sidechain_indices, :]
            bb_gt   = coords_3d[:, backbone_indices, :]
            sc_gt   = coords_3d[:, sidechain_indices, :]

            bb_mse = criterion(bb_pred, bb_gt)
            sc_mse = criterion(sc_pred, sc_gt)
            loss   = bb_mse + sc_mse

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_bb_mse += bb_mse.item()
            total_train_sc_mse += sc_mse.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_bb   = total_train_bb_mse / len(train_loader)
        avg_train_sc   = total_train_sc_mse / len(train_loader)

        model.eval()
        test_loss_val = 0.0
        test_bb_val   = 0.0
        test_sc_val   = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                B_times_N, _ = data.x.shape
                B = B_times_N // num_atoms
                pred_bb = backbone_decoder(data.x, z_ref=z_ref.to(device), log_debug=False)
                full_pred = model(data.x, pred_bb, z_ref=z_ref.to(device), log_debug=False)

                coords_3d = data.y.view(B, num_atoms, 3)
                bb_pred = full_pred[:, backbone_indices, :]
                sc_pred = full_pred[:, sidechain_indices, :]
                bb_gt   = coords_3d[:, backbone_indices, :]
                sc_gt   = coords_3d[:, sidechain_indices, :]

                bb_mse = criterion(bb_pred, bb_gt)
                sc_mse = criterion(sc_pred, sc_gt)
                val_loss = bb_mse + sc_mse

                test_loss_val += val_loss.item()
                test_bb_val   += bb_mse.item()
                test_sc_val   += sc_mse.item()

        avg_test_loss = test_loss_val / len(test_loader)
        avg_test_bb   = test_bb_val / len(test_loader)
        avg_test_sc   = test_sc_val / len(test_loader)

        if (epoch+1) % config.get("log_interval", 10) == 0 or (epoch+1)==epochs:
            logger.info(f"[SidechainDecoder] Epoch {epoch+1}/{epochs} => "
                        f"TRAIN_LOSS: {avg_train_loss:.4f} (BB={avg_train_bb:.4f}, SC={avg_train_sc:.4f}), "
                        f"TEST_LOSS: {avg_test_loss:.4f} (BB={avg_test_bb:.4f}, SC={avg_test_sc:.4f})")

        if (epoch+1) % config.get("save_interval", 10) == 0:
            checkpoint_state = {
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_state, checkpoint_path, logger)

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    logger.info(f"Sidechain decoder checkpoint saved to {checkpoint_path}")
    return model

#################################################################
# (K) Output Generation
#################################################################
def export_final_outputs(raw_dataset, dec_dataset, hno_model, backbone_decoder, sidechain_decoder,
                           z_ref, backbone_indices, sidechain_indices, num_atoms, struct_dir, latent_dir,
                           use_diff=False, diff_bb=None, diff_sc=None):
    """
    Exports final outputs:
      (1) Normal coords:
          - HNO => hno_reconstructions.h5
          - Backbone => backbone_coords.h5
          - Full => full_coords.h5
      (2) If use_diff==True and diff_bb, diff_sc are valid => also export
          - backbone_coords_diff => backbone_coords_diff.h5
          - full_coords_diff => full_coords_diff.h5
    """
    logger.info("Exporting final outputs: HNO, backbone-only coords, full protein.")

    hno_model.eval()
    backbone_decoder.eval()
    sidechain_decoder.eval()

    # Define output paths
    hno_recon_path       = os.path.join(struct_dir, "hno_reconstructions.h5")
    backbone_path        = os.path.join(struct_dir, "backbone_coords.h5")
    full_path            = os.path.join(struct_dir, "full_coords.h5")

    # Potential diffusion paths
    backbone_path_diff   = os.path.join(struct_dir, "backbone_coords_diff.h5")
    full_path_diff       = os.path.join(struct_dir, "full_coords_diff.h5")

    total_samples = len(raw_dataset)
    logger.info(f"Number of samples for final export: {total_samples}")

    backbone_count  = len(backbone_indices)
    sidechain_count = len(sidechain_indices)

    with h5py.File(hno_recon_path, "w") as hno_h5, \
         h5py.File(backbone_path, "w") as bb_h5, \
         h5py.File(full_path, "w") as full_h5:

        dset_hno  = hno_h5.create_dataset("hno_coords", (total_samples, num_atoms, 3), dtype='float32')
        dset_bb   = bb_h5.create_dataset("backbone_coords", (total_samples, backbone_count, 3), dtype='float32')
        dset_full = full_h5.create_dataset("full_coords", (total_samples, num_atoms, 3), dtype='float32')

        for idx in range(total_samples):
            raw_data = raw_dataset[idx].to(device)
            dec_data = dec_dataset[idx].to(device)

            with torch.no_grad():
                # (a) HNO coords
                recon = hno_model(raw_data.x, raw_data.edge_index, log_debug=False)
                dset_hno[idx, :, :] = recon.cpu().numpy()

                # (b) Normal backbone coords
                hno_latent = dec_data.x
                pred_bb = backbone_decoder(hno_latent, z_ref=z_ref, log_debug=False)
                dset_bb[idx, :, :] = pred_bb.cpu().numpy()

                # (c) Normal full coords
                full_pred = sidechain_decoder(hno_latent, pred_bb, z_ref=z_ref, log_debug=False)
                dset_full[idx, :, :] = full_pred.cpu().numpy()

    logger.info(f"Saved normal outputs to: \n  {hno_recon_path}\n  {backbone_path}\n  {full_path}")

    # -------------------------
    # If user wants diffusion-based coords:
    #   We'll produce backbone_coords_diff.h5 and full_coords_diff.h5
    #   But we must ensure that we have enough diff_bb, diff_sc entries.
    # -------------------------
    if use_diff and diff_bb is not None and diff_sc is not None:
        logger.info("Also exporting coords with DIFFUSED pooled embeddings override.")
        N_diff = min(len(dec_dataset), diff_bb.shape[0], diff_sc.shape[0])
        logger.info(f"We have N_diff={N_diff} samples to export with diffused embeddings.")

        with h5py.File(backbone_path_diff, "w") as bb_diff_h5, \
             h5py.File(full_path_diff, "w") as full_diff_h5:

            dset_bb_diff   = bb_diff_h5.create_dataset("backbone_coords_diff", (N_diff, backbone_count, 3), dtype='float32')
            dset_full_diff = full_diff_h5.create_dataset("full_coords_diff", (N_diff, num_atoms, 3), dtype='float32')

            for idx in range(N_diff):
                dec_data = dec_dataset[idx].to(device)
                hno_latent = dec_data.x
                B_times_N, E = hno_latent.shape
                B = B_times_N // num_atoms
                if B != 1:
                    raise ValueError("This diffusion export logic assumes 1 sample per dataset entry. Found B>1?")

                # slice out the single row of diffused embeddings
                # shape => [1, pool_dim]
                bb_over = diff_bb[idx:idx+1, :].to(device)
                sc_over = diff_sc[idx:idx+1, :].to(device)

                # override the normal pooling
                with torch.no_grad():
                    pred_bb_diff = backbone_decoder(hno_latent, z_ref=z_ref,
                                                    override_pooled_backbone=bb_over)
                    full_pred_diff = sidechain_decoder(hno_latent, pred_bb_diff, z_ref=z_ref,
                                                       override_pooled_sidechain=sc_over)

                dset_bb_diff[idx, :, :]   = pred_bb_diff.cpu().numpy()
                dset_full_diff[idx, :, :] = full_pred_diff.cpu().numpy()

        logger.info(f"Saved diffusion-based coords to:\n  {backbone_path_diff}\n  {full_path_diff}")
    else:
        logger.info("No diffusion override used or diff arrays not provided; skipping diff output.")


#################################################################
# (J) Main
#################################################################
def main():
    logger.info(f"Using device: {device}")
    out_dirs = config.get("output_directories", {})
    ckpt_dir = out_dirs.get("checkpoint_dir", "checkpoints")
    struct_dir = out_dirs.get("structure_dir", "structures")
    latent_dir = out_dirs.get("latent_dir", "latent_reps")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    json_path = config["json_path"]
    pdb_filename = config.get("pdb_filename", "heavy_chain.pdb")

    coords_per_frame = load_heavy_atom_coords_from_json(json_path, logger)
    logger.info(f"Parsing PDB: {pdb_filename}")
    _, atoms_in_order = parse_pdb(pdb_filename)
    renumbered_dict = renumber_atoms_and_residues(atoms_in_order)
    backbone_indices, sidechain_indices = get_global_indices(renumbered_dict)
    logger.info(f"Found {len(backbone_indices)} backbone and {len(sidechain_indices)} sidechain atoms.")

    coords_aligned = align_frames_to_first(coords_per_frame, logger)
    num_atoms = coords_aligned[0].shape[0]
    logger.info(f"Each frame contains {num_atoms} heavy atoms.")

    knn_value = config["knn_value"]
    dataset = build_graph_dataset(coords_aligned, knn_neighbors=knn_value, logger=logger)

    # 1) HNO
    hno_conf = config["hno_training"]
    train_data_hno, test_data_hno = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader_hno = DataLoader(train_data_hno, batch_size=hno_conf["batch_size"], shuffle=True)
    test_loader_hno  = DataLoader(test_data_hno,  batch_size=hno_conf["batch_size"], shuffle=False)

    cheb_order = config["cheb_order"]
    hidden_dim = config["hidden_dim"]
    hno_model = HNO(hidden_dim, K=cheb_order)
    hno_ckpt = os.path.join(ckpt_dir, config.get("hno_ckpt", "hno_model.pth"))

    logger.info(f"Training/loading HNO => {hno_conf['num_epochs']} epochs, LR={hno_conf['learning_rate']}")
    hno_model = train_hno_model(
        model=hno_model,
        train_loader=train_loader_hno,
        test_loader=test_loader_hno,
        num_epochs=hno_conf["num_epochs"],
        learning_rate=hno_conf["learning_rate"],
        checkpoint_path=hno_ckpt,
        save_interval=hno_conf.get("save_interval", 10)
    )

    # 2) Build dec_dataset from HNO latents
    logger.info("Building dataset for decoders using HNO latent embeddings...")
    hno_model.eval()
    dec_dataset = []
    with torch.no_grad():
        for data in dataset:
            x_emb = hno_model.forward_representation(data.x.to(device), data.edge_index.to(device))
            dec_data = Data(x=x_emb.cpu(), y=data.y, edge_index=data.edge_index, batch=data.batch)
            dec_dataset.append(dec_data)

    # 3) Split again for decoders
    train_data_dec, test_data_dec = train_test_split(dec_dataset, test_size=0.1, random_state=42)
    train_loader_dec = DataLoader(train_data_dec, batch_size=16, shuffle=True)
    test_loader_dec  = DataLoader(test_data_dec,  batch_size=16, shuffle=False)

    # 4) z_ref from first aligned frame
    logger.info("Computing z_ref from first aligned frame.")
    with torch.no_grad():
        X_ref = coords_aligned[0].to(device)
        edge_ref = knn_graph(X_ref, k=knn_value, loop=False)
        X_ref_emb = hno_model.forward_representation(X_ref, edge_ref, log_debug=True)
        z_ref = X_ref_emb
        logger.debug(f"z_ref => {z_ref.shape}")

    # 5) Backbone decoder
    stepA_conf = config["decoderB_training"]
    pooling_dim_backbone = tuple(config.get("pooling_dim_backbone", [20,4]))
    backbone_decoder_ckpt = os.path.join(ckpt_dir, "decoder_backbone.pth")

    backbone_decoder_model = BackboneDecoder(
        num_total_atoms=num_atoms,
        backbone_indices=backbone_indices,
        emb_dim=hidden_dim,
        pooling_dim=pooling_dim_backbone,
        mlp_depth=stepA_conf.get("decoder_depth", 2),
    ).to(device)

    logger.info("Training/loading backbone decoder.")
    backbone_decoder_model = train_backbone_decoder(
        model=backbone_decoder_model,
        train_loader=train_loader_dec,
        test_loader=test_loader_dec,
        device=device,
        logger=logger,
        config=stepA_conf,
        checkpoint_path=backbone_decoder_ckpt,
        z_ref=z_ref
    )

    # 6) Sidechain decoder
    stepB_conf = config["decoderSC_training"]
    pooling_dim_sidechain = tuple(config.get("pooling_dim_sidechain", [20,4]))
    sidechain_decoder_ckpt = os.path.join(ckpt_dir, "decoder_sidechain.pth")

    sidechain_decoder_model = SidechainDecoder(
        num_total_atoms=num_atoms,
        sidechain_indices=sidechain_indices,
        backbone_indices=backbone_indices,
        emb_dim=hidden_dim,
        pooling_dim=pooling_dim_sidechain,
        mlp_depth=stepB_conf.get("decoder_depth", 2),
        arch_type=stepB_conf.get("arch_type", 0)
    ).to(device)

    logger.info("Training/loading sidechain decoder.")
    sidechain_decoder_model = train_sidechain_decoder(
        model=sidechain_decoder_model,
        train_loader=train_loader_dec,
        test_loader=test_loader_dec,
        backbone_decoder=backbone_decoder_model,
        device=device,
        logger=logger,
        config=stepB_conf,
        checkpoint_path=sidechain_decoder_ckpt,
        z_ref=z_ref
    )

    logger.info("All training tasks completed successfully!")

    # 7) Optionally load diffused embeddings for final export
    diff_bb_torch = None
    diff_sc_torch = None
    if args.use_diffusion:
        if not args.diffused_backbone_h5 or not args.diffused_sidechain_h5:
            logger.warning("use_diffusion=True but missing either diffused_backbone_h5 or diffused_sidechain_h5. Skipping diffusion override.")
        else:
            logger.info("Loading diffused pooled embeddings to override final structure generation.")
            with h5py.File(args.diffused_backbone_h5, "r") as f:
                diff_bb_np = f["generated_diffusion"][:]
            with h5py.File(args.diffused_sidechain_h5, "r") as f:
                diff_sc_np = f["generated_diffusion"][:] 
            print(diff_bb_np.ndim, diff_bb_np.shape)
            print(diff_sc_np.ndim, diff_sc_np.shape)

            # Flatten if shape is [N,H,W]
            if diff_bb_np.ndim == 3:
                N,H,W = diff_bb_np.shape
                diff_bb_np = diff_bb_np.reshape(N,H*W)
            if diff_sc_np.ndim == 3:
                N,H,W = diff_sc_np.shape
                diff_sc_np = diff_sc_np.reshape(N,H*W)

            # Flatten if shape is [N,1,H,W]
            if diff_bb_np.ndim == 4 and diff_bb_np.shape[1] == 1:
                N,H,W = diff_bb_np.shape
                diff_bb_np = diff_bb_np.reshape(N,H*W)
            if diff_sc_np.ndim == 4 and diff_sc_np.shape[1] == 1:
                N,H,W = diff_sc_np.shape
                diff_sc_np = diff_sc_np.reshape(N,H*W)

            print(diff_bb_np.ndim, diff_bb_np.shape)
            print(diff_sc_np.ndim, diff_sc_np.shape)

            diff_bb_torch = torch.from_numpy(diff_bb_np).float()
            diff_sc_torch = torch.from_numpy(diff_sc_np).float()
            logger.info(f"diff_bb_torch => {diff_bb_torch.shape}, diff_sc_torch => {diff_sc_torch.shape}")

    # 8) Export final outputs
    export_final_outputs(
        raw_dataset=dataset,
        dec_dataset=dec_dataset,
        hno_model=hno_model,
        backbone_decoder=backbone_decoder_model,
        sidechain_decoder=sidechain_decoder_model,
        z_ref=z_ref,
        backbone_indices=backbone_indices,
        sidechain_indices=sidechain_indices,
        num_atoms=num_atoms,
        struct_dir=struct_dir,
        latent_dir=latent_dir,
        use_diff=args.use_diffusion and (diff_bb_torch is not None) and (diff_sc_torch is not None),
        diff_bb=diff_bb_torch,
        diff_sc=diff_sc_torch
    )

    sys.stdout.flush()

if __name__ == "__main__":
    main()