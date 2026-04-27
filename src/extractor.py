from transformers import AutoModel, AutoTokenizer

import torch
import h5py
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
def get_cls_token(sentences, model, tokenizer, device, task_name, split):
    base = Path(__file__).resolve()
    route = base.parent.parent / 'data' / 'cls_tokens' / f'cls_tokens_{task_name}_{split}.h5'
    
    route.parent.mkdir(parents=True, exist_ok=True) # Create directory if doesn't exist
    
    # If .h5 file already exist
    if route.exists():
        print(f'File {route} already exists.')
        return None
    
    all_cls = [] # Get CLS token for each sentence on sentences
        
    # Create file .h5 only if doesn't exist
    with h5py.File(route, 'x') as f:
        for idx, sentence in enumerate(sentences):
            inputs = tokenizer(sentence, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get all layers
            cls_all_layers = np.stack([hs[0, 0, :].cpu().numpy() for hs in outputs.hidden_states])
            all_cls.append(cls_all_layers)
            
            # Save in .h5 file
            ds = f.create_dataset(f'sentence_{idx}', data=cls_all_layers)
            ds.attrs['text'] = sentence # Save sentence as metadata
            
    print(f'CLS tokens successfully stored in {route}.')
    
    return np.array(all_cls)