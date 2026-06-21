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
    def __init__(self, n_roles, role_dim, filler_vocab_size, filler_dim, output_dim, pretrained_filler_weight=None):
        super().__init__()

        # Role embedding (trainable)
        self.role_embedding = nn.Embedding(n_roles, role_dim)

        # Filler embedding: frozen pretrained word embeddings + trainable linear projection
        self.filler_embedding = nn.Embedding(filler_vocab_size, filler_dim)
        
        if pretrained_filler_weight is not None:
            self.filler_embedding.weight.data = pretrained_filler_weight.float()
            
        self.filler_embedding.weight.requires_grad = False  # freeze, like McCoy/Jawahar
        self.filler_projection = nn.Linear(filler_dim, filler_dim)

        self.projection = nn.Linear(in_features=filler_dim * role_dim, out_features=output_dim)

    def forward(self, filler_ids, role_ids, attention_mask):
        # Static filler representation: frozen embed -> learned linear projection
        fillers = self.filler_embedding(filler_ids)
        fillers = self.filler_projection(fillers)

        role_vecs = self.role_embedding(role_ids)

        # Mask padding on BOTH fillers and roles so padded positions contribute 0
        mask_expanded = attention_mask.unsqueeze(-1).float()
        fillers_masked = fillers * mask_expanded
        role_vecs_masked = role_vecs * mask_expanded

        # Tensor product, then flatten
        tpr_sum = torch.einsum('bwf, bwr -> bfr', fillers_masked, role_vecs_masked)
        tpr_flat = tpr_sum.flatten(start_dim=1)

        output = self.projection(tpr_flat)

        return output
    
# Dataset for TPDN
class TPDNDataset(Dataset):
    def __init__(self, filler_ids_list, role_ids_list, targets):
        assert len(filler_ids_list) == len(role_ids_list) == len(targets), f'Corpus mismatch: {len(filler_ids_list)} fillers, {len(role_ids_list)} roles, {len(targets)} targets.'

        self.filler_ids_list = filler_ids_list
        self.role_ids_list = role_ids_list
        self.targets = targets

    def __len__(self):
        return len(self.filler_ids_list)

    def __getitem__(self, idx):
        filler_id = self.filler_ids_list[idx]
        role_id = self.role_ids_list[idx]
        target = self.targets[idx]

        filler_tensor = torch.tensor(filler_id, dtype=torch.long)
        role_tensor = torch.tensor(role_id, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return {
                'filler_ids': filler_tensor,
                'role_ids': role_tensor,
                'targets': target_tensor
            }

# Custom function for TPDNDataset
def tpdn_collate_fn(batch):
    filler_ids_list = [item['filler_ids'] for item in batch]
    role_ids_list = [item['role_ids'] for item in batch]
    targets_list = [item['targets'] for item in batch]

    lengths = torch.tensor([len(r) for r in role_ids_list])

    # Pad sequences (filler id 0 and role id 0 are padding; masked out in forward)
    filler_ids_padded = pad_sequence(filler_ids_list, batch_first=True, padding_value=0)
    role_ids_padded = pad_sequence(role_ids_list, batch_first=True, padding_value=0)

    batch_size = len(batch)
    w_max = role_ids_padded.shape[1]

    # Real tokens marked with 1, padding with 0
    attention_mask = torch.arange(w_max).expand(batch_size, w_max) < lengths.unsqueeze(1)
    attention_mask = attention_mask.float()

    targets_stacked = torch.stack(targets_list)

    return {
        'filler_ids': filler_ids_padded,
        'role_ids': role_ids_padded,
        'attention_mask': attention_mask,
        'targets': targets_stacked
    }
    
# Function to train TPDN (with early stopping on a validation set)
def train_tpdn(tpdn, dataloader, device, val_loader=None, n_epochs=100, lr=1e-3, patience=10):
    import copy

    tpdn = tpdn.to(device)
    optimizer = Adam([p for p in tpdn.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = copy.deepcopy(tpdn.state_dict())
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        tpdn.train()
        batch_iterator = tqdm(dataloader, desc=f'Epoch {epoch + 1} / {n_epochs}', leave=False)
        for batch in batch_iterator:
            filler_ids = batch['filler_ids'].to(device, dtype=torch.long)
            role_ids = batch['role_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.float32)
            targets = batch['targets'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            preds = tpdn(filler_ids, role_ids, attention_mask)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            batch_iterator.set_postfix({'loss': f'{loss.item():.4f}'})

        # Early stopping on validation MSE
        if val_loader is not None:
            val_mse = evaluate_mse(tpdn, val_loader, device)
            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(tpdn.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    # Restore best checkpoint (by validation loss)
    if val_loader is not None:
        tpdn.load_state_dict(best_state)

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
            filler_ids = batch['filler_ids'].to(device, dtype=torch.long)
            role_ids = batch['role_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.float32)
            targets = batch['targets'].to(device, dtype=torch.float32)
 
            preds = tpdn(filler_ids, role_ids, attention_mask)
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
def run_tpdn_evaluation(filler_ids_train, filler_ids_test, targets_train, targets_test,
                        sentences_train, sentences_test,
                        role_schemes_dict, build_vocab_fn, map_roles_fn,
                        n_fillers, pretrained_filler_weight=None,
                        seeds=[42, 43, 44, 45, 46], role_dim=20, filler_dim=768,
                        n_epochs=100, lr=1e-3, batch_size=32, patience=10,
                        val_ratio=0.1, device='cuda',
                        output_filename='tpdn_table4_results'):
    # Config
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    n_layers = targets_train.shape[1]
    if pretrained_filler_weight is not None:
        filler_dim = pretrained_filler_weight.shape[1]
 
    results_flat = []
 
    # Carve out a fixed validation split from train (for early stopping)
    n_train_total = len(filler_ids_train)
    n_val = max(1, int(n_train_total * val_ratio))
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_train_total)
    val_idx = set(perm[:n_val].tolist())
    tr_idx = [i for i in range(n_train_total) if i not in val_idx]
    va_idx = [i for i in range(n_train_total) if i in val_idx]
 
    # Loop over role schemes
    for scheme_name, scheme_fn in role_schemes_dict.items():
        print(f'Processing scheme: {scheme_name}')
 
        # Generate raw roles
        raw_roles_train = [scheme_fn(sent) for sent in sentences_train]
        raw_roles_test = [scheme_fn(sent) for sent in sentences_test]
 
        # Build role vocabulary over train+test
        role_to_id = build_vocab_fn(raw_roles_train + raw_roles_test)
        n_roles = len(role_to_id)
 
        # Map raw roles to numerical IDs
        role_ids_train_all = map_roles_fn(raw_roles_train, role_to_id)
        role_ids_test = map_roles_fn(raw_roles_test, role_to_id)
 
        # Loop over BERT layers
        for layer_idx in range(n_layers):
            # Slice targets for the specific layer
            layer_targets_train_all = targets_train[:, layer_idx, :]
            layer_targets_test = targets_test[:, layer_idx, :]
 
            # Split train into train/val (by fixed indices)
            f_tr = [filler_ids_train[i] for i in tr_idx]
            r_tr = [role_ids_train_all[i] for i in tr_idx]
            t_tr = layer_targets_train_all[tr_idx]
 
            f_va = [filler_ids_train[i] for i in va_idx]
            r_va = [role_ids_train_all[i] for i in va_idx]
            t_va = layer_targets_train_all[va_idx]
 
            train_dataset = TPDNDataset(f_tr, r_tr, t_tr)
            val_dataset = TPDNDataset(f_va, r_va, t_va)
            test_dataset = TPDNDataset(filler_ids_test, role_ids_test, layer_targets_test)
 
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tpdn_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=tpdn_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tpdn_collate_fn)
 
            layer_seed_mses = []
 
            # Loop over seeds
            for seed in seeds:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
 
                model = TPDN(
                    n_roles=n_roles,
                    role_dim=role_dim,
                    filler_vocab_size=n_fillers,
                    filler_dim=filler_dim,
                    output_dim=targets_train.shape[2],
                    pretrained_filler_weight=pretrained_filler_weight
                )
 
                model = train_tpdn(model, train_loader, device, val_loader=val_loader,
                                   n_epochs=n_epochs, lr=lr, patience=patience)
 
                seed_mse = evaluate_mse(model, test_loader, device)
                layer_seed_mses.append(seed_mse)
 
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