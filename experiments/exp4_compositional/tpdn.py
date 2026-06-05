import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

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

# Function to train TPDN
def train_tpdn(tpdn, dataloader, device, n_epochs=10, lr=1e-3):
    # Config
    tpdn = tpdn.to(device) 
    tpdn.train()
    
    optimizer = Adam(tpdn.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        batch_iterator = tqdm(dataloader, desc=f'Epoch {epoch + 1} / {n_epochs}')
        for batch in batch_iterator:
            fillers = batch['fillers'].to(device, dtype=torch.float32)
            role_ids = batch['role_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.float32)
            targets = batch['targets'].to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            preds = tpdn(fillers, role_ids, attention_mask)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            batch_iterator.set_postfix({'loss' : f'{loss.item():.4f}'})
        
    return tpdn