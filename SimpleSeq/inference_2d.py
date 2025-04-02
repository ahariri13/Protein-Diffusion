
from  HNOSimpleSeq import HNOSimpleSeq
### import arguments from the main file
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
num_residues = 2191

@torch.no_grad()
def inference_with_pooled(
    model: HNOSimpleSeq,
    dataRef,                # Some new reference graph data
    pooled_sidechain_atoms,      # [batch_size, some_dim]
    pooled_backbone_atoms,  # [batch_size, some_dim]
    device="cuda"
):
    """
    Use only convRef, convRef2, and mlpRep2 on new pooled inputs.
    """
    repeat= pooled_backbone_atoms.shape[0]
    model.eval()  # ensure we are in eval mode

    ## Concatenate x_main and x_backbone to dim 1
    full_pooling=torch.cat((pooled_sidechain_atoms,pooled_backbone_atoms),dim=1)
    
    ### Flatten 
    full_pooling=full_pooling.reshape(repeat,-1)
    
    ### Embed x_ref
    # with torch.no_grad():
    x_ref=dataRef.x
    edgeref=dataRef.edge_index
    for convref in model.convRef:
        # x = conv(x, edgeref)
        # if convref != model.convs[-1]:
        x_ref = convref(x_ref, edgeref)
        x_ref = F.tanh(x_ref)
        x_ref = nn.BatchNorm1d(x_ref.size(1)).to(device)(x_ref)
        x = F.dropout(x_ref, p=0.15, training=model.training)

    # Step 1: Repeat x_ref 100 times along the batch dimension
    x_ref = x_ref.repeat(pooled_backbone_atoms.shape[0], 1, 1).to(device)   # Shape: [100, 2208, 64]

    # Step 2: Expand B to match the shape of A_repeated for concatenation
    backbone_repeat = pooled_backbone_atoms.view(pooled_backbone_atoms.shape[0],-1).unsqueeze(1).repeat(1, num_residues, 1).to(device)  # Shape: [100, 2208, 64]

    # Step 3: Concatenate A_repeated and B_expanded along the last dimension
    phase1 = torch.cat((x_ref, backbone_repeat), dim=-1).to(device)   # Shape: [100, 2208, 128]

    #### Phase 1 
    phase1=model.mlp(phase1)
    # phase1=F.relu(phase1)
    # phase1=model.mlp2(phase1)
    #### Phase 1  done

    schain_repeat=pooled_sidechain_atoms.view(pooled_sidechain_atoms.shape[0],-1).unsqueeze(1).repeat(1, num_residues, 1).to(device)
    # print("schain_repeat: "+str(schain_repeat.shape))

    phase2 = torch.cat((phase1, schain_repeat), dim=-1).to(device)   # Shape: [100, 2207, 128]

    phase2=model.mlpPhase2(phase2)
    # phase2=F.relu(phase2)
    # phase2=model.mlpPhase22(phase2)

    result=phase2.reshape(-1,phase2.shape[2])

    result=model.mlpRep2(result)

    return result.reshape(-1,num_residues,3)

