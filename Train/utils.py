import torch


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

def get_ith_frame_graph(ind,residue_numbers,residues_data):
   ### Append frame of each element of residue data in heave atom coords
  atom_types=[]
  coords=[]
  is_backbone=[]
  for res_num_str in residue_numbers:
      residue = residues_data[res_num_str]
      ith_frame_coordinates = residue['heavy_atom_coords_per_frame'][ind]

      for atom in residue['heavy_atoms']:
          atom_types.append(atom['name'])
          is_backbone.append(atom['is_backbone'])
      coords+=ith_frame_coordinates

  return atom_types, torch.Tensor(coords), is_backbone

from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_graph

def protein_to_graph(coords,atom_type,is_backbone):
  edge_index=knn_graph(coords, k=8, batch=None, loop=False)
  graph = Data(x=coords, edge_index=edge_index,atom_type=atom_type,backbone=is_backbone)
  return graph