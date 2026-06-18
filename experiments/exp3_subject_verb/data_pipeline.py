import requests
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from scripts.generate_dataset_sva_es import run_ancora_sva_pipeline

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
    
    sva_data = sva_data.sample(frac=1, random_state=7).reset_index(drop=True) # Shuffle data
    
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

# Function to load data for BETO
def load_and_split_data_es(train_size=0.05, valid_size=0.05):
    base = Path(__file__).resolve()
    data_route = base.parent.parent.parent / 'generated_data' / 'ancora_sva_es' / 'dataset_sva.csv'
    
    if not data_route.exists():
        print(f'The file doesn\'t exist in {data_route}.')
        print(f'Generating data...')
        run_ancora_sva_pipeline()
        
    # Here ever you get the file
    sva_data = pd.read_csv(data_route)
    
    # The data generated have max_lenght 50  
    
    sva_data = sva_data.sample(frac=1, random_state=7).reset_index(drop=True) # Shuffle data
    
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
def create_binary_labels(datasets, lang):
    if lang == 'en':
        mapping = {
            'VBZ': 0, # Singular
            'VBP': 1 # Plural
        }
        source_col = 'verb_pos'
    elif lang == 'es':
        mapping = {
            'Sing': 0,
            'Plur': 1
        }
        source_col = 'verb_number'
    
    for split_name in datasets:
        df = datasets[split_name]
        df['label'] = df[source_col].map(mapping)
        
    return datasets

#
def align_and_mask_datasets(datasets, tokenizer, lang):
    if lang == 'en':
        sent_col, vidx_col, offset = 'orig_sentence', 'verb_index', 1
    elif lang == 'es':
        sent_col, vidx_col, offset = 'sentence', 'verb_pos', 0
    
    for split_name in datasets:
        df = datasets[split_name]
        
        input_ids_list = []
        token_idx_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {split_name}'):
            # Get words and verb idx
            words = row[sent_col].split()
            v_idx = int(row[vidx_col]) - offset
            
            # Tokenization
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=True,
                add_special_tokens=True
            )
            
            word_ids = encoding.word_ids()
            verb_token_idx = [i for i, wid in enumerate(word_ids) if wid == v_idx]
            
            input_ids = encoding['input_ids']
            for i in verb_token_idx:
                input_ids[i] = tokenizer.mask_token_id
            
            # Save data
            input_ids_list.append(input_ids)
            token_idx_list.append(verb_token_idx[0]) # Standard probing practice
            
        # Add results in dataframe
        df['bert_input_ids'] = input_ids_list
        df['verb_token_index'] = token_idx_list
        
    return datasets

# Function to group test df into num of attractors
def create_difficulty_buckets(test_df):
    buckets = {}
    pure_df = test_df[test_df['n_intervening'] == test_df['n_diff_intervening']].copy()
    
    for n in range(5):
        if n < 4:
            # Buckets for 0, 1, 2, 3
            buckets[f'{n}'] = pure_df[pure_df['n_diff_intervening'] == n].copy()
        else:
            # Bucket for 4 or more
            buckets['4+'] = pure_df[pure_df['n_diff_intervening'] >= n].copy()
    
    return buckets

def run_full_pipeline(tokenizer, lang):
    if lang == 'en':
        download_sva_dataset()
        datasets = load_and_split_data(train_size=0.09, valid_size=0.01) # Values as Jawahar et al. 2019 propose
    elif lang == 'es':
        datasets = load_and_split_data_es(train_size=0.09, valid_size=0.01)
        
    datasets = create_binary_labels(datasets)
    datasets = align_and_mask_datasets(datasets, tokenizer)
    
    # Reset index
    for split in ['train', 'valid', 'test']:
        datasets[split] = datasets[split].reset_index(drop=True)
        
    test_buckets = create_difficulty_buckets(datasets['test'])
    
    return datasets, test_buckets