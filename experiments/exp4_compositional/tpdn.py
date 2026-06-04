import torch
import torch.nn as nn

class TPDN(nn.Module):
    def __init__(self, n_roles, role_dim, filler_dim, output_dim):
        super().__init__()
        
        self.role_embedding = nn.Embedding(n_roles, role_dim)
        self.projection = nn.Linear(in_features=filler_dim*role_dim, out_features=output_dim)
        
    def forward(self, fillers, role_ids, attention_mask):
        role_vecs = self.role_embedding(role_ids)
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        fillers_masked = fillers*mask_expanded
        
        tpr_sum = torch.einsum('bwf, bwr -> bfr', fillers_masked, role_vecs)
        tpr_flat = tpr_sum.flatten(start_dim=1)
        
        output = self.projection(tpr_flat)
        
        return output