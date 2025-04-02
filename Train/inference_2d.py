
from  HNO import HNO
### import arguments from the main file
import torch
import torch.nn.functional as F
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
num_residues = 2191


@torch.no_grad()
def inference_with_pooled(
    model: HNO,
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

    ### Embed Xref
    # with torch.no_grad():
    x_ref=dataRef.x
    edgeref=dataRef.edge_index
    for conv2 in model.convRef:
        # if conv2 != self.convs[-1]:
        x_ref = conv2(x_ref, edgeref)
        x_ref = F.tanh(x_ref)
        x_ref = nn.BatchNorm1d(x_ref.size(1)).to(device)(x_ref)
        x_ref = F.dropout(x_ref, p=0.15, training=model.training)


    # Step 1: Repeat x_ref 100 times along the batch dimension
    x_ref = x_ref.repeat(repeat, 1, 1).to(device)   # Shape: [100, 2208, 64]

    full_pooling = full_pooling.unsqueeze(1).repeat(1, num_residues, 1).to(device)  # Shape: [100, 2208, 64]

    result = torch.cat((x_ref, full_pooling), dim=-1).to(device)   # Shape: [100, 2208, 128]

    result=result.reshape(-1,result.shape[2])

    result=model.mlpRep2(result)

    return result.reshape(-1,num_residues,3)

"""
dataRef = torch.load("dataRef.pt")

### Create random tensors 
pooled_main_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)
pooled_backbone_atoms  = torch.randn(5,args.window_size, args.hidden).to(device)

result = inference_with_pooled(
    model_inference,
    dataRef,
    pooled_main_atoms=pooled_main_atoms,
    pooled_backbone_atoms=pooled_backbone_atoms,
    device=device
)
"""