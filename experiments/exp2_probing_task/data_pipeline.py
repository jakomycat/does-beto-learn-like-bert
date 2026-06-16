import requests
from pathlib import Path

# Function to download X-PROBE
def download_xprobe_data(task_name, lang):
    splits = ['tr', 'va', 'te']
    
    # See if data/xprobe exist
    base = Path(__file__).resolve()
    xprobe_route = base.parent.parent.parent / 'data' / 'xprobe' / lang / task_name
    
    xprobe_route.mkdir(parents=True, exist_ok=True) # Create directories if they don't exist
    
    for split in splits:
        xprobe_route_task = xprobe_route / f'{split}.txt'
        
        if xprobe_route_task.exists():
            print(f'The file {split}.txt for task {task_name} ({lang}) already exists.')
            continue
        
        url = f'https://raw.githubusercontent.com/ltgoslo/xprobe/master/{lang}/{task_name}/{split}.txt'
        
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                
                with open(xprobe_route_task, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            print(f'The file {split}.txt for {task_name} ({lang}) was downloaded successfully.')  
            
        # Exception handling
        except requests.exceptions.HTTPError:
            print(f'Split \'{split}\' for task \'{task_name}\' in \'{lang}\' does not exist in X-PROBE.')

        except requests.exceptions.RequestException as e:
            print(f'Network error while downloading \'{split}\' for \'{task_name}\': {e}')
        
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
    download_seteval_data(task_name)
    
    # Read data
    train, test, validation = read_senteval_file(task_name)
    
    # Get data encoded
    (train_enc, test_enc, val_enc), dicts = encode_labels(train['labels'], test['labels'], validation['labels'])
    
    all_data_packaged = {
        'train': {'sentences': train['data'], 'labels': train_enc},
        'val': {'sentences': validation['data'], 'labels': val_enc},
        'test': {'sentences': test['data'], 'labels': test_enc},
        'dicts': dicts
    }
    
    return all_data_packaged