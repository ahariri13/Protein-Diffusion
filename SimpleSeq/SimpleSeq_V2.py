import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,global_mean_pool, ChebConv,global_add_pool

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
num_residues= 2191#len(residues_data) ### Instead of num_residues or 274 in the Ca case 

class SimpleSeqV2(torch.nn.Module):

    def __init__(self,hiddens_dims,K,window_size,num_layers,mlp_layers):
        super(SimpleSeqV2, self).__init__()

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

        self.mlpRep2 = nn.ModuleList([
            nn.Linear(hiddens_dims, hiddens_dims),  # 1st Linear      
            nn.Linear(hiddens_dims, hiddens_dims), # 2nd Linear
            nn.Linear(hiddens_dims, 3)  # 3rd Linear (output)
        ])

        self.convRef = torch.nn.ModuleList()
        self.convRef.append(ChebConv(3, hiddens_dims,K=K))
        for _ in range(2):
            self.convRef.append(ChebConv(hiddens_dims, hiddens_dims,K=K))


    def forward(self, x,edge_index,batch,mask,dataRef):

        for conv in self.convs:
            x = conv(x, edge_index)
            ## Add batchnorm and relu except at last layer
            if conv != self.convs[-1]:
                x = F.tanh(x)
                # x = nn.BatchNorm1d(x.size(1)).to(device)(x)
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
        x_backbone_atoms=x_backbone_atoms.reshape(len(batch.unique()),-1,x.shape[1])
        x_schain_atoms=x_schain_atoms.reshape(len(batch.unique()),-1,x.shape[1])

        x_backbone_atoms= x_backbone_atoms.mean(dim=1).unsqueeze(1)   #self.pool(x_backbone_atoms.unsqueeze(0)).squeeze(0)
        x_schain_atoms=  x_schain_atoms.mean(dim=1).unsqueeze(1)     #self.pool(x_schain_atoms.unsqueeze(0)).squeeze(0)

        

        ### Embed x_ref
        # with torch.no_grad():
        x_ref=dataRef.x
        edgeref=dataRef.edge_index
        for conv2 in self.convRef:
            if conv2 != self.convs[-1]:
                x_ref = conv2(x_ref, edgeref)
                x_ref = F.tanh(x_ref)
                # x_ref = nn.BatchNorm1d(x_ref.size(1)).to(device)(x_ref)
                x_ref = F.dropout(x_ref, p=0.15, training=self.training)

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

        for lin in self.mlpRep2:
            result=lin(result)
            result=F.relu(result)
            result=F.dropout(result, p=0.2, training=self.training)

        return result.view(-1,3),x_backbone_atoms, x_schain_atoms