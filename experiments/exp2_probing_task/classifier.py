import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import copy

# Single-layer perceptron
class ProbeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ProbeClassifier, self).__init__()
        self.layer = nn.Linear(input_dim, num_classes) # Unique layer
        
    def forward(self, X):
        output = self.layer(X)
        return output
    
# Prepare data
def create_dataloader(X, y, batch_size=32, is_train=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train
    )
    
    return dataloader

# Loop of train
def train_probe(model, train_loader, val_loader, device, max_epochs=100, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Variable for Early Stopping
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    
    epoch_iterator = tqdm(range(max_epochs), desc='Training Probe')
    
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad() # Clean mathematic memory
            
            prediction = model(batch_X)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Only calculate the loss
                prediction = model(batch_X)
                loss = criterion(prediction, batch_y)
        
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Detect if the model improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else: # The model didn't improve
            epochs_without_improvement += 1
            
        # Update progress
        epoch_iterator.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Loss': f'{avg_val_loss:.4f}',
            'Patience': f'{epochs_without_improvement}/{patience}'
        })
    
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
    # Load the best weights
    model.load_state_dict(best_model_weights)
    
    return model

#
def evaluate_layer(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, num_classes, device):
    train_loader = create_dataloader(X_train, y_train, batch_size=32, is_train=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=32, is_train=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=32, is_train=False)
    
    model = ProbeClassifier(input_dim, num_classes).to(device)
    
    # Model trained with Early Stopping
    model = train_probe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=100,
        patience=5
    )
    
    # Evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
    # Calculate accuracy and return this
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy

def evaluate_all_layers(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, device, task_name, output_filename):
    num_layers = X_train.shape[1]
    accuracies = []

    for layer_idx in range(num_layers):
        # Extract the specific layer for each split
        X_train_layer = X_train[:, layer_idx, :]
        X_val_layer = X_val[:, layer_idx, :]
        X_test_layer = X_test[:, layer_idx, :]
        
        input_dim = X_train_layer.shape[1]
        
        accuracy = evaluate_layer(
            X_train_layer,
            y_train,
            X_val_layer,
            y_val,
            X_test_layer,
            y_test,
            input_dim,
            num_classes,
            device
        )
        
        accuracies.append(accuracy)

    # Save results to CSV
    base = Path(__file__).resolve()
    csv_route = base.parent.parent.parent / 'results' / 'csv' / f'{output_filename}.csv'
    csv_route.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CSV or create new DataFrame
    if csv_route.exists():
        df = pd.read_csv(csv_route, index_col=0)
    else:
        df = pd.DataFrame()

    # Add/overwrite column for this task
    df[task_name] = accuracies
    df.index.name = 'layer'

    df.to_csv(csv_route)
    print(f'Results saved to {csv_route}.')

    return accuracies