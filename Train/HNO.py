# Install required packages.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool, ChebConv,global_add_pool, TopKPooling

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
num_residues= 2191      #len(residues_data) ### Instead of num_residues or 274 in the Ca case 


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


# from torch_scatter import scatter
class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x

class HNO(torch.nn.Module):

    def __init__(self,hiddens_dims,K,window_size,num_layers):
        super(HNO, self).__init__()

        # self.conv1 = ChebConv(3, hiddens_dims,K=K)

        self.convs = torch.nn.ModuleList()

        self.convs.append(ChebConv(3, hiddens_dims,K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hiddens_dims, hiddens_dims,K=K))

        self.bano1 = torch.nn.BatchNorm1d(num_features= hiddens_dims)
        self.bano2 = torch.nn.BatchNorm1d(num_features= hiddens_dims)
        self.bano3 = torch.nn.BatchNorm1d(num_features= hiddens_dims)
        self.banoDec = torch.nn.BatchNorm1d(num_features= hiddens_dims)

        self.pool = nn.AdaptiveAvgPool2d((window_size,hiddens_dims))

        # self.mlpRep1 = nn.Linear(2*int(window_size*hiddens_dims)+hiddens_dims,hiddens_dims)  # MLP(640, 3, nlayer=2, with_final_activation=False)
        # self.convDec = ChebConv(2*int(m*hiddens_dims)+hiddens_dims, hiddens_dims,K=4)  # MLP(640, 3, nlayer=2, with_final_activation=False)
        # self.convDec = ChebConv(hiddens_dims, hiddens_dims,K=4)  # MLP(640, 3, nlayer=2, with_final_activation=False)
        # self.convDec2 = ChebConv(hiddens_dims, hiddens_dims,K=4)  # MLP(640, 3, nlayer=2, with_final_activation=False)
        # self.mlpRep2 = MLP(hiddens_dims, 3, nlayer=3, with_final_activation=False)

        self.mlpRep2 = MLP(2*int(window_size*hiddens_dims)+hiddens_dims, 3, nlayer=3, with_final_activation=False)

        self.convRef = torch.nn.ModuleList()
        self.convRef.append(ChebConv(3, hiddens_dims,K=K))
        for _ in range(num_layers):
            self.convRef.append(ChebConv(hiddens_dims, hiddens_dims,K=K))


    def forward(self, x,edge_index,batch,mask,dataRef):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.tanh(x)
            x = nn.BatchNorm1d(x.size(1)).to(device)(x)
            x = F.dropout(x, p=0.1, training=self.training)

        # Convert the mask to a tensor (if not already)
        mask_tensor = torch.tensor(mask).reshape(-1)

        # Get indices where mask is False  (atoms) and True (backbones)
        false_indices = torch.nonzero(~mask_tensor)
        true_indices = torch.nonzero(mask_tensor)


        ### Extract main atoms and side chains from mask
        x_main_atoms=x[false_indices].reshape(-1,x.shape[1])
        x_backbone_atoms=x[true_indices].reshape(-1,x.shape[1])

        ### Reshape to bs, num_atoms,hidden
        x_main_atoms=x_main_atoms.reshape(len(batch.unique()),-1,x.shape[1])
        x_backbone_atoms=x_backbone_atoms.reshape(len(batch.unique()),-1,x.shape[1])

        x_main_atoms=self.pool(x_main_atoms.unsqueeze(0)).squeeze(0)
        x_backbone_atoms=self.pool(x_backbone_atoms.unsqueeze(0)).squeeze(0)

        ## Concatenate x_main and x_backbone to dim 1
        full_pooling=torch.cat((x_main_atoms,x_backbone_atoms),dim=1)
        # full_pooling=  x_main_atoms+x_backbone_atoms
        
        ### Flatten 
        full_pooling=full_pooling.reshape(len(batch.unique()),-1)
        

        ### Embed Xref
        # with torch.no_grad():
        x_ref=dataRef.x
        edgeref=dataRef.edge_index
        for conv2 in self.convRef:
            # if conv2 != self.convs[-1]:
            x_ref = conv2(x_ref, edgeref)
            x_ref = F.tanh(x_ref)
            x_ref = nn.BatchNorm1d(x_ref.size(1)).to(device)(x_ref)
            x_ref = F.dropout(x_ref, p=0.15, training=self.training)


        # Step 1: Repeat x_ref 100 times along the batch dimension
        x_ref = x_ref.repeat(len(batch.unique()), 1, 1).to(device)   # Shape: [100, 2208, 64]

        # Step 2: Expand B to match the shape of A_repeated for concatenation
        full_pooling = full_pooling.unsqueeze(1).repeat(1, num_residues, 1).to(device)  # Shape: [100, 2208, 64]

        # Step 3: Concatenate A_repeated and B_expanded along the last dimension
        result = torch.cat((x_ref, full_pooling), dim=-1).to(device)   # Shape: [100, 2208, 128]

        result=result.reshape(-1,result.shape[2])

        result=self.mlpRep2(result)

        return result.view(-1,3),x_backbone_atoms, x_main_atoms