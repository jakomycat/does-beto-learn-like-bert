import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from pathlib import Path

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
    
# Dataset for TPDN
class TPDNDataset(Dataset):
    def __init__(self, fillers_list, role_ids_list, targets):
        assert len(fillers_list) == len(role_ids_list) == len(targets), f'Corpus mismatch: {len(fillers_list)} fillers, {len(role_ids_list)} roles, {len(targets)} targets.'
        
        self.fillers_list = fillers_list
        self.role_ids_list = role_ids_list
        self.targets = targets
        
    def __len__(self):
        return len(self.fillers_list)
    
    def __getitem__(self, idx):
        filler = self.fillers_list[idx]
        role_id = self.role_ids_list[idx]
        target = self.targets[idx]
        
        filler_tensor = torch.tensor(filler, dtype=torch.float32)
        role_tensor = torch.tensor(role_id, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return {
                'fillers': filler_tensor,
                'role_ids': role_tensor,
                'targets': target_tensor
            }

# Custom function for TPDNDataset
def tpdn_collate_fn(batch):
    fillers_list = [item['fillers'] for item in batch]
    role_ids_list = [item['role_ids'] for item in batch]
    targets_list = [item['targets'] for item in batch]
    
    lengths = torch.tensor([len(r) for r in role_ids_list])
    
    # Fill the vectors
    fillers_padded = pad_sequence(fillers_list, batch_first=True, padding_value=0.0)
    role_ids_padded = pad_sequence(role_ids_list, batch_first=True, padding_value=0)
    
    batch_size = len(batch)
    w_max = role_ids_padded.shape[1]
    
    # Here important words are marked with a 1, otherwise a 0
    attention_mask = torch.arange(w_max).expand(batch_size, w_max) < lengths.unsqueeze(1)
    attention_mask = attention_mask.float()
    
    targets_stacked = torch.stack(targets_list)
    
    return {
        'fillers': fillers_padded,
        'role_ids': role_ids_padded,
        'attention_mask': attention_mask,
        'targets': targets_stacked
    }
    
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

# Function to calculate MSE
def evaluate_mse(tpdn, dataloader, device):
    tpdn.eval()
    
    criterion = nn.MSELoss()
    
    total_error_weighted = 0.0
    total_sentences = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract components
            fillers = batch['fillers'].to(device, dtype=torch.float32)
            role_ids = batch['role_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.float32)
            targets = batch['targets'].to(device, dtype=torch.float32)
            
            preds = tpdn(fillers, role_ids, attention_mask)
            loss = criterion(preds, targets)
            
            current_batch_size = targets.size(0)
            
            total_error_weighted += loss.item() * current_batch_size
            total_sentences += current_batch_size
        
    global_mse = total_error_weighted / total_sentences
    
    return global_mse

# Function to map raw text/tuple roles to vocabulary integer IDs
def map_roles_to_ids(raw_roles_list, role_to_id):
    # OOV roles fall back to the reserved <unk> id (0 if present, else 0).
    unk_id = role_to_id.get('<unk>', 0)
    mapped_corpus = []
    for sentence_roles in raw_roles_list:

        mapped_sentence = [role_to_id.get(role, unk_id) for role in sentence_roles]
        mapped_corpus.append(mapped_sentence)

    return mapped_corpus

# Function to run the full evaluation across schemes, layers, and seeds
def run_tpdn_evaluation(fillers_train, fillers_test, targets_train, targets_test, sentences_train, sentences_test,
                        role_schemes_dict, build_vocab_fn, map_roles_fn,
                        seeds=[42, 43, 44, 45, 46], role_dim=20, n_epochs=10, lr=1e-3, batch_size=32, device='cuda',
                        output_filename='tpdn_table4_results'):
    # Config
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    n_layers = targets_train.shape[1]
    filler_dim = fillers_train[0].shape[1]
    
    results_flat = []
    
    # Loop over role schemes
    for scheme_name, scheme_fn in role_schemes_dict.items():
        print(f'Processing scheme: {scheme_name}')
        
        # Generate raw roles
        raw_roles_train = [scheme_fn(sent) for sent in sentences_train]
        raw_roles_test = [scheme_fn(sent) for sent in sentences_test]
        
        # Build vocabulary using train set
        role_to_id = build_vocab_fn(raw_roles_train)
        n_roles = len(role_to_id)
        
        # Map raw roles to numerical IDs
        role_ids_train = map_roles_fn(raw_roles_train, role_to_id)
        role_ids_test = map_roles_fn(raw_roles_test, role_to_id)
        
        # Loop over BERT layers
        for layer_idx in range(n_layers):
            # Slice targets for the specific layer
            layer_targets_train = targets_train[:, layer_idx, :]
            layer_targets_test = targets_test[:, layer_idx, :]
            
            # Create Datasets
            train_dataset = TPDNDataset(fillers_train, role_ids_train, layer_targets_train)
            test_dataset = TPDNDataset(fillers_test, role_ids_test, layer_targets_test)
            
            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=tpdn_collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=tpdn_collate_fn
            )
            
            layer_seed_mses = []
            
            # Loop over seeds
            for seed in seeds:
                # Set seed for reproducibility
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    
                # Instantiate model
                model = TPDN(
                    n_roles=n_roles,
                    role_dim=role_dim,
                    filler_dim=filler_dim,
                    output_dim=filler_dim
                )
                
                # Train model
                model = train_tpdn(model, train_loader, device, n_epochs=n_epochs, lr=lr)
                
                # Evaluate MSE
                seed_mse = evaluate_mse(model, test_loader, device)
                layer_seed_mses.append(seed_mse)
                
            # Calculate mean and std over seeds
            mean_mse = np.mean(layer_seed_mses)
            std_mse = np.std(layer_seed_mses)
            
            results_flat.append({
                'scheme': scheme_name,
                'layer': layer_idx,
                'mse_mean': mean_mse,
                'mse_std': std_mse
            })
            
    # Convert to DataFrame and pivot to get Table 4 layout
    df_flat = pd.DataFrame(results_flat)
    df_table4 = df_flat.pivot(index='layer', columns='scheme', values='mse_mean')
    
    # Save results to CSV
    base_path = Path(__file__).resolve().parents[2]
    output_dir = base_path / 'results' / 'csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f'{output_filename}.csv'
    
    # If csv exists, load and combine data
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path, index_col='layer')
        df_table4 = df_table4.combine_first(df_existing)
    
    df_table4.to_csv(csv_path)
    print(f'Results saved/updated to {csv_path}')
    
    return df_table4