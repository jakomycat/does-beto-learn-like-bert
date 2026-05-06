from transformers import AutoModel, AutoTokenizer

import torch
import h5py
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Function to get pre-trained model
def load_model_and_tokenizer(lang, device):
    # To BERT
    if lang == 'en':
        model_name = 'bert-base-uncased'
        
    # To BETO
    elif lang == 'es':
        raise ValueError('Its not implemented')
    
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.to(device)
    
    return model, tokenizer

# Function to get span representation (obviously)
def get_span_representation(span_samples, model, tokenizer, device):
    representations = []
    
    for sample in span_samples:
        label = sample['label']
        sentence = sample['sentence']
        idx_start = sample['start']
        idx_end = sample['end']
        
        # Tokenize
        inputs = tokenizer(
            sentence.split(),
            return_tensors='pt',
            is_split_into_words=True
        )
        
        # Map word indices to BERT token indices
        word_ids = inputs.word_ids()
        bert_start = word_ids.index(idx_start)
        bert_end = len(word_ids) - 1 - word_ids[::-1].index(idx_end)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            
        layer_representations = []
        
        # Get hidden states for each layer - without the embeddings
        for hidden_states in outputs.hidden_states:
            hidden_states = hidden_states.squeeze(0)
            
            # Get first and last hidden state
            h_first = hidden_states[bert_start] # Exclude [CLS]
            h_final = hidden_states[bert_end] # Exclude [SEP]
            
            # Get element-wise product and difference
            product = h_first * h_final
            difference = h_first - h_final
            
            # Concatenate
            span_representation = torch.cat([h_first, h_final, product, difference], dim=0)
            layer_representations.append(span_representation.cpu().numpy())
        
        representations.append({
            'representation': layer_representations,
            'label': label
        })
        
    return representations

# Function to get [CLS] token
def get_cls_token(sentences, model, tokenizer, device, task_name, split, batch_size=32):
    base = Path(__file__).resolve()
    route = base.parent.parent / 'data' / 'cls_tokens' / f'cls_tokens_{task_name}_{split}.h5'

    route.parent.mkdir(parents=True, exist_ok=True)

    # If .h5 file already exists, load and return
    if route.exists():
        with h5py.File(route, 'r') as f:
            return f['cls_tokens'][:]

    # Get dimensions from model config
    # n_layers = num_hidden_layers + 1
    n_layers    = getattr(model.config, 'num_hidden_layers',
                  getattr(model.config, 'num_layers', 12)) + 1
    hidden_size = model.config.hidden_size
    n_sentences = len(sentences)

    # Create .h5 file only if it doesn't exist
    with h5py.File(route, 'x') as f:

        # Pre-allocate dataset — one single contiguous block
        ds = f.create_dataset(
            'cls_tokens',
            shape=(n_sentences, n_layers, hidden_size),
            dtype=np.float16,
            compression='lzf',
            chunks=(min(batch_size, n_sentences), n_layers, hidden_size)
        )

        # Save sentences as metadata
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('sentences', data=np.array(sentences, dtype=dt))

        # Process sentences in batches
        for i in tqdm(range(0, n_sentences, batch_size), desc='Extracting CLS tokens'):
            batch_sentences = sentences[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract [CLS] token from each layer and stack
            batch_cls = torch.stack([
                hs[:, 0, :] for hs in outputs.hidden_states
            ], dim=1).half().cpu().numpy()

            # Write batch to disk
            ds[i : i + len(batch_sentences)] = batch_cls

    print(f'CLS tokens successfully stored in {route}.')

    # Read from disk and return
    with h5py.File(route, 'r') as f:
        return f['cls_tokens'][:]
    
# Function to extract the [Mask] token vectors from each layer
def extract_verb_features(sentences, verb_idx, model, tokenizer, device, split, batch_size=32):
    base = Path(__file__).resolve()
    route = base.parent.parent / 'data' / 'features' / f'sva_features_{split}.h5'
    route.parent.mkdir(parents=True, exist_ok=True)

    if route.exists():
        with h5py.File(route, 'r') as f:
            return f['verb_outputs'][:]

    n_layers = getattr(model.config, 'num_hidden_layers', 12) + 1
    hidden_size = model.config.hidden_size
    n_sentences = len(sentences)

    with h5py.File(route, 'x') as f:
        ds = f.create_dataset(
            'verb_outputs',
            shape=(n_sentences, n_layers, hidden_size),
            dtype=np.float16,
            compression='lzf'
        )

        model.eval()
        for i in tqdm(range(0, n_sentences, batch_size), desc=f'Processing {split}'):
            batch_sentences = sentences[i : i + batch_size]
            batch_idx = verb_idx[i : i + batch_size]
            
            inputs = tokenizer(
                batch_sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            batch_range = torch.arange(len(batch_sentences), device=device)
            target_idx = torch.tensor(batch_idx, device=device)

            # Extract [Mask] token
            batch_features = torch.stack([
                hs[batch_range, target_idx, :] for hs in outputs.hidden_states
            ], dim=1).half().cpu().numpy()

            ds[i : i + len(batch_sentences)] = batch_features

    print(f'SVA vectors avalaible at: {route}')
    
    with h5py.File(route, 'r') as f:
        return f['verb_outputs'][:]