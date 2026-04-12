from transformers import AutoModel, AutoTokenizer
import torch

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