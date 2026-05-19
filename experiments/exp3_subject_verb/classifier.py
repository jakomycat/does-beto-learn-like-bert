import torch

from sklearn.metrics import accuracy_score

from pathlib import Path
import pandas as pd
import h5py
import numpy as np

from experiments.exp2_probing_task.classifier import ProbeClassifier, create_dataloader, train_probe # To reuse code

def evaluate_by_buckets(model, X_test_layer, y_test, buckets, device):
    model.eval()
    bucket_results = {}
    
    for b_name, b_df in buckets.items():
        # Get bucket idx for the current difficulty
        filtered_df = b_df[b_df['n_intervening'] == b_df['n_diff_intervening']]
        idx = filtered_df.index.tolist()
        
        # Filter data 
        X_bucket = torch.tensor(X_test_layer[idx], dtype=torch.float32, device=device)
        y_bucket = torch.tensor(y_test[idx], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(X_bucket)
            y_preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(y_bucket.cpu(), y_preds.cpu())
            bucket_results[f"bucket_{b_name}"] = acc
            
    return bucket_results

def run_full_sva_evaluation(X_train_path, y_train, X_val_path, y_val, X_test_path, y_test, buckets, num_classes, device, output_filename):
    # It's only to know how many layers has the model
    with h5py.File(X_train_path, 'r') as f:
        _, num_layers, hidden_size = f['verb_outputs'].shape
    
    all_layer_data = []

    for layer_idx in range(num_layers):
        print(f'Current layer: {layer_idx}')
        
        with h5py.File(X_train_path, 'r') as f_train, h5py.File(X_val_path, 'r') as f_val, h5py.File(X_test_path, 'r') as f_test:
            X_train_l = f_train['verb_outputs'][:, layer_idx, :]
            X_val_l = f_val['verb_outputs'][:, layer_idx, :]
            X_test_l = f_test['verb_outputs'][:, layer_idx, :]
        
        train_loader = create_dataloader(X_train_l, y_train, device, is_train=True)
        val_loader = create_dataloader(X_val_l, y_val, device, is_train=False)
        
        # Model train
        model = ProbeClassifier(hidden_size, num_classes).to(device)
        model = train_probe(model, train_loader, val_loader)
        
        # Evaluation
        row_results = {'layer': layer_idx}
        bucket_metrics = evaluate_by_buckets(model, X_test_l, y_test, buckets, device)
        row_results.update(bucket_metrics)
        
        all_layer_data.append(row_results)
        
        del X_train_l, X_val_l, X_test_l, train_loader, val_loader, model # Free up GPU memory

    # Save data
    df_results = pd.DataFrame(all_layer_data)
    
    base = Path(__file__).resolve()
    csv_route = base.parent.parent.parent / 'results' / 'csv' / f'{output_filename}.csv'
    csv_route.parent.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(csv_route, index=False)
    print(f'Results saved to {csv_route}.')
    
    return df_results