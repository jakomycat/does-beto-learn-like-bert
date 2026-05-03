import requests
from pathlib import Path

import pandas as pd

# Function to dowload dataset from dropbox
def download_sva_dataset():
    url = 'https://www.dropbox.com/scl/fi/fph8xsz683bne62rpxcjs/agr_50_mostcommon_10K.tsv.gz?e=3&rlkey=t7tgb87xd3073qzv7fghu7vm6&dl=1'
    file_name = 'agr_50_mostcommon_10K.tsv.gz'
    
    # Get route
    base = Path(__file__).resolve()
    route = base.parent.parent.parent / 'data' / 'SVADataset'
    
    route.mkdir(parents=True, exist_ok=True)
    
    route_file = route / file_name
    
    if route_file.exists():
        print(f'The file {file_name} already exists.')
        return
    
    try:
        request = requests.get(url, stream=True)
        request.raise_for_status() # Check if the data was download
        
        with open(route_file, 'wb') as file:
            # Write file in chunks from 8 KB
            for chunk in request.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    
        print(f'The file {file_name} was downloaded successful.')
        
    # Downloading error exception
    except requests.exceptions.RequestException as e:
        print(f'Network error while downloading \'{file_name}\': {e}')
        
# Function to load and split data
def load_and_split_data(train_size=0.05, valid_size=0.05, max_seq_len=50):
    base = Path(__file__).resolve()
    route = base.parent.parent.parent / 'data' / 'SVADataset' / 'agr_50_mostcommon_10K.tsv.gz'
    
    if not route.exists():
        print(f'The file doesn\'t exist in {route}.')
        return None
    
    sva_data = pd.read_csv(route, sep='\t') # Get dataset
    
    # Filter by lenght
    sva_data['word_count'] = sva_data['orig_sentence'].str.split().str.len()
    sva_data = sva_data[sva_data['word_count'] <= max_seq_len].copy()
    
    sva_data = sva_data.sample(frac=1, random_state=123).reset_index(drop=True) # Shuffle data
    
    # Split
    num_total = len(sva_data)
    num_train = int(num_total * train_size)
    num_valid = int(num_total * valid_size)
    
    train_df = sva_data.iloc[0:num_train]
    valid_df = sva_data.iloc[num_train:num_train + num_valid]
    test_df = sva_data.iloc[num_train + num_valid:]
    
    split_data = {
        'train': train_df,
        'valid': valid_df,
        'test': test_df
    }
    
    return split_data
        
# Function to classify verb in singular or plural
def create_binary_labels(datasets):
    pos_to_label = {
        'VBZ': 0, # Singular
        'VBP': 1 # Plural
    }
    
    for split_name in datasets:
        df = datasets[split_name]
        df['label'] = df['verb_pos'].map(pos_to_label)
        
    return datasets