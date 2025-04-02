import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,global_mean_pool, ChebConv,global_add_pool

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

class HNOSimpleSeq(torch.nn.Module):

    def __init__(self,hiddens_dims,K,window_size,num_layers,mlp_layers):
        super(HNOSimpleSeq, self).__init__()

        # self.conv1 = ChebConv(3, hiddens_dims,K=K)

        self.convs = torch.nn.ModuleList()

        self.convs.append(ChebConv(3, hiddens_dims,K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hiddens_dims, hiddens_dims,K=K))

        self.pool = nn.AdaptiveAvgPool2d((window_size,hiddens_dims))

        self.mlpRep1 = nn.Linear(2*int(window_size*hiddens_dims)+hiddens_dims,hiddens_dims)  
        self.mlp = nn.Linear((window_size+1)*hiddens_dims,hiddens_dims)  
        self.mlp2 = nn.Linear(hiddens_dims,hiddens_dims)  

        self.mlpPhase2 = nn.Linear((window_size+1)*hiddens_dims,hiddens_dims)  
        self.mlpPhase22 = nn.Linear(hiddens_dims,hiddens_dims)

        self.mlpRep2 = MLP(hiddens_dims, 3, nlayer= mlp_layers, with_final_activation=False)

        self.convRef = torch.nn.ModuleList()
        self.convRef.append(ChebConv(3, hiddens_dims,K=K))
        for _ in range(num_layers):
            self.convRef.append(ChebConv(hiddens_dims, hiddens_dims,K=K))
            
    def forward(self, x,edge_index,batch,mask,dataRef):

        for conv in self.convs:
            x = conv(x, edge_index)
            # if conv != self.convs[-1]:
            x = nn.BatchNorm1d(x.size(1)).to(device)(x)
            x = F.tanh(x)
            x = nn.BatchNorm1d(x.size(1)).to(device)(x)
            x = F.dropout(x, p=0.15, training=self.training)
            

        # Convert the mask to a tensor (if not already)
        mask_tensor = torch.tensor(mask).reshape(-1)

        # Get indices where mask is False  (atoms) and True (backbones)
        false_indices = torch.nonzero(~mask_tensor)
        true_indices = torch.nonzero(mask_tensor)


        ### Extract main atoms and side chains from mask
        x_schain_atoms=x[false_indices].reshape(-1,x.shape[1])
        x_backbone_atoms=x[true_indices].reshape(-1,x.shape[1])

        ### Reshape to bs, num_atoms,hidden
        x_schain_atoms=x_schain_atoms.reshape(len(batch.unique()),-1,x.shape[1])
        x_backbone_atoms=x_backbone_atoms.reshape(len(batch.unique()),-1,x.shape[1])

        x_schain_atoms=self.pool(x_schain_atoms.unsqueeze(0)).squeeze(0)
        x_backbone_atoms=self.pool(x_backbone_atoms.unsqueeze(0)).squeeze(0)
        

        ### Embed x_ref
        # with torch.no_grad():
        x_ref=dataRef.x
        edgeref=dataRef.edge_index
        for conv2 in self.convRef:
            # if conv2 != self.convs[-1]:
            x_ref = conv2(x_ref, edgeref)
            x_ref = F.tanh(x_ref)
            x_ref = nn.BatchNorm1d(x_ref.size(1)).to(device)(x_ref)
            # x_ref = F.dropout(x_ref, p=0.15, training=self.training)

        # Step 1: Repeat x_ref 100 times along the batch dimension
        x_ref = x_ref.repeat(len(batch.unique()), 1, 1).to(device)   # Shape: [100, 2208, 64]

        # Step 2: Expand B to match the shape of A_repeated for concatenation
        backbone_repeat = x_backbone_atoms.view(x_backbone_atoms.shape[0],-1).unsqueeze(1).repeat(1, num_residues, 1).to(device)  # Shape: [100, 2208, 64]

        # Step 3: Concatenate A_repeated and B_expanded along the last dimension
        phase1 = torch.cat((x_ref, backbone_repeat), dim=-1).to(device)   # Shape: [100, 2208, 128]

        #### Phase 1 
        phase1=self.mlp(phase1)
        # phase1=F.relu(phase1)
        # phase1=self.mlp2(phase1)
        #### Phase 1  done

        schain_repeat=x_schain_atoms.view(x_schain_atoms.shape[0],-1).unsqueeze(1).repeat(1, num_residues, 1).to(device)
        # print("schain_repeat: "+str(schain_repeat.shape))

        phase2 = torch.cat((phase1, schain_repeat), dim=-1).to(device)   # Shape: [100, 2207, 128]

        phase2=self.mlpPhase2(phase2)
        # phase2=F.relu(phase2)
        # phase2=self.mlpPhase22(phase2)

        result=phase2.reshape(-1,phase2.shape[2])

        result=self.mlpRep2(result)

        return result.view(-1,3),x_backbone_atoms, x_schain_atoms