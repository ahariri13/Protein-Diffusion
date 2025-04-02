#!/usr/bin/env python3
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

import argparse
import h5py
import torch
import numpy as np
import random

def load_h5_structure(h5file, key):
    with h5py.File(h5file, "r") as f:
        if key not in f:
            avail_keys = list(f.keys())
            raise KeyError(f"Key '{key}' not found in '{h5file}'. Available keys: {avail_keys}")
        data = f[key][:]
    return torch.from_numpy(data).float()

def load_xref(xref_file):
    arr = np.load(xref_file)
    return torch.from_numpy(arr).float()

def parse_backbone_indices_from_pdb(pdb_file):
    allowed = {"N", "CA", "C", "O", "OXT"}
    backbone_indices = []
    current_idx = 0
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name in allowed:
                    backbone_indices.append(current_idx)
                current_idx += 1
            else:
                pass
    return sorted(backbone_indices)

def kabsch_alignment_torch(P, Q):
    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    C = torch.mm(Q_centered.t(), P_centered)
    U, S, Vt = torch.svd(C)
    if torch.det(torch.mm(U, Vt)) < 0:
        U[:, -1] = -U[:, -1]
    R = torch.mm(U, Vt)
    Q_aligned = torch.mm(Q_centered, R) + centroid_P
    return Q_aligned

def calculate_lddt_global_torch(ref_coords, model_coords, cutoff=15.0, seq_sep=2, thresholds=[0.5, 1.0, 2.0, 4.0]):
    device = ref_coords.device
    L = ref_coords.shape[0]
    diff_ref = ref_coords.unsqueeze(0) - ref_coords.unsqueeze(1)
    ref_dists = torch.norm(diff_ref, dim=2)
    diff_model = model_coords.unsqueeze(0) - model_coords.unsqueeze(1)
    model_dists = torch.norm(diff_model, dim=2)

    idx = torch.arange(L, device=device)
    seq_sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= seq_sep
    cutoff_mask = ref_dists <= cutoff
    valid_mask = seq_sep_mask & cutoff_mask

    diff_matrix = torch.abs(model_dists - ref_dists).unsqueeze(-1)  # (L,L,1)
    thr_tensor = torch.tensor(thresholds, device=device).view(1,1,-1)
    hits = (diff_matrix < thr_tensor).float()  # (L,L,T)
    hit_scores = hits.sum(dim=-1) / float(len(thresholds))

    per_atom_lddt = torch.zeros(L, device=device)
    for i in range(L):
        valid = valid_mask[i]
        if valid.sum() > 0:
            per_atom_lddt[i] = hit_scores[i][valid].mean()
        else:
            per_atom_lddt[i] = 0.0

    overall_lddt = per_atom_lddt.mean().item()
    return overall_lddt, per_atom_lddt

def main():
    parser = argparse.ArgumentParser(
        description="Compute lDDT (all-atom or backbone-only) with optional sampling of models."
    )
    parser.add_argument("--h5file", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--xref", required=True, help="Path to X_ref.npy (all heavy-atom coords).")
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--seq_sep", type=int, default=2)

    parser.add_argument("--backbone_only", action="store_true",
                        help="If set, do backbone-only lDDT. Must pass --pdb to get backbone indices.")
    parser.add_argument("--pdb", type=str, default=None,
                        help="PDB file used to extract backbone indices if --backbone_only is set.")
    # Sampling:
    parser.add_argument("--max_samples", type=int, default=100,
                        help="If >0, randomly sample that many models from the dataset.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set random seed for reproducibility if sampling.")

    args = parser.parse_args()

    # Load reference
    ref_all = load_xref(args.xref)  # shape (L_all,3)
    print(f"Loaded reference from {args.xref}, shape={tuple(ref_all.shape)}")

    # If backbone-only, parse indices from PDB => slice ref
    if args.backbone_only:
        if not args.pdb:
            raise ValueError("Must provide --pdb when using --backbone_only.")
        backbone_indices = parse_backbone_indices_from_pdb(args.pdb)
        ref_coords = ref_all[backbone_indices, :]
        print(f"Backbone-only mode: using {len(backbone_indices)} backbone atoms from PDB={args.pdb}")
    else:
        backbone_indices = None
        ref_coords = ref_all

    # Load predicted coords
    pred_coords = load_h5_structure(args.h5file, args.key)  # shape (N, L,3)
    N_total = pred_coords.shape[0]
    print(f"Loaded predicted coords from {args.h5file}, key={args.key}, shape={pred_coords.shape}")

    # If all-atom: must match dimension
    if not args.backbone_only:
        if pred_coords.shape[1] != ref_all.shape[0]:
            raise ValueError(f"Pred shape[1]={pred_coords.shape[1]} != ref_all.shape[0]={ref_all.shape[0]}!")
    else:
        # backbone-only => must have at least max(backbone_indices).
        max_idx = max(backbone_indices)
        if pred_coords.shape[1] <= max_idx:
            raise ValueError(f"Pred coords have {pred_coords.shape[1]} atoms; but backbone indices go up to {max_idx}!")
        # The reference is shape (L_bb,3) now.

    # If sampling:
    if args.max_samples > 0 and args.max_samples < N_total:
        if args.seed is not None:
            random.seed(args.seed)
        subset = random.sample(range(N_total), args.max_samples)
        subset = sorted(subset)
    else:
        subset = range(N_total)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_coords = ref_coords.to(device)
    pred_coords = pred_coords.to(device)

    lddt_values = []
    for count, i_model in enumerate(subset, start=1):
        model_xyz = pred_coords[i_model]
        if args.backbone_only:
            model_xyz = model_xyz[backbone_indices, :]
        # Align
        aligned = kabsch_alignment_torch(ref_coords, model_xyz)
        # lDDT
        val, _ = calculate_lddt_global_torch(ref_coords, aligned,
                                             cutoff=args.cutoff,
                                             seq_sep=args.seq_sep)
        lddt_values.append(val)

    # Print metrics
    arr_np = np.array(lddt_values)
    mean_val = arr_np.mean()
    std_val  = arr_np.std()#*args.max_samples
    print("\n=====================================================")
    print(f"Number of models processed = {len(subset)} (out of {N_total})")
    print(f"lDDT mean = {mean_val:.4f}, stdev = {std_val:.4f}")

if __name__ == "__main__":
    main()