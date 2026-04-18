import requests
from pathlib import Path

# Function to download data from original GitHub
def download_seteval_data(task_name):
    # See if data/SentEval exist
    base = Path(__file__).resolve()
    senteval_route = base.parent.parent.parent / 'data' / 'SentEval'
    
    senteval_route.mkdir(parents=True, exist_ok=True) # Create directories if they don't exist
    
    # See if task data exist
    senteval_route_task = senteval_route / f'{task_name}.txt'
    
    if senteval_route_task.exists():
        print(f'The file {task_name}.txt already exists.')
        return
    
    url = f'https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/probing/{task_name}.txt'
    
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            with open(senteval_route_task, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): # Chunks of 8KB
                    if chunk: f.write(chunk) # Get all data
                    
        print(f'The file {task_name}.txt was downloaded successful.')
    
    # Exception handling
    except requests.exceptions.HTTPError as e:
        print(f'Task \'{task_name}\' does not exist in SentEval.')

    except requests.exceptions.RequestException as e:
        print(f'Network error while downloading \'{task_name}\': {e}')
        
# Function to read SentEval's file and get train, test, validation
def read_senteval_file(task_name):
    base = Path(__file__).resolve()
    senteval_route = base.parent.parent.parent / 'data' / 'SentEval' / f'{task_name}.txt'
    
    train, test, validation = {'data':[], 'labels':[]}, {'data':[], 'labels':[]}, {'data':[], 'labels':[]}
    with open(senteval_route, 'r', encoding='utf-8') as f:
        for line in f:
            split_type, label, text = line.split('\t') # This return a list with 3 elements
            
            if split_type == 'tr':
                train['data'].append(text.replace('\n', ''))
                train['labels'].append(label)
                
            elif split_type == 'va':
                validation['data'].append(text.replace('\n', ''))
                validation['labels'].append(label)
                
            elif split_type == 'te':
                test['data'].append(text.replace('\n', ''))
                test['labels'].append(label)
    
    return train, test, validation

# Function to
def encode_labels(train_labels, test_labels, validation_labels):
    unique_labels = sorted(list(set(train_labels))) # Get every unique label
    
    label_to_id = {label:i for i, label in enumerate(unique_labels)}
    id_to_label = {i:label for i, label in enumerate(unique_labels)}

    train_encoded = [label_to_id[label] for label in train_labels]
    test_encoded = [label_to_id[label] for label in test_labels]
    val_encoded = [label_to_id[label] for label in validation_labels]
        
    return (train_encoded, test_encoded, val_encoded), (label_to_id, id_to_label)

# Function to run pipeline
def load_probing_task(task_name):
    # Verify if data exist
    download_seteval_data('past_present')
    
    # Read data
    train, test, validation = read_senteval_file('past_present')
    
    # Get data encoded
    all_data_packaged = encode_labels(train['labels'], test['labels'], validation['labels'])
    
    return all_data_packaged