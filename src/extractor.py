from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to get pre-trained model
def load_model_and_tokenizer(lang, device):
    # To BERT
    if lang == 'en':
        model_name = 'bert-base-cased'
        
    # To BETO
    elif lang == 'es':
        raise ValueError('Its not implemented')
    
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.to(device)
    
    return model, tokenizer

# Function to get span representation (obviously)
def get_span_representation(span_samples, model, tokenizer):
    representations = []
    
    for sample in span_samples:
        text = sample['text']
        label = sample['label']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt')
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            
        layer_representations = []
        
        # Get hidden states for each layer - without the embeddings
        for hidden_states in outputs.hidden_states[1:-1]:
            hidden_states = outputs.last_hidden_state.squeeze(0)
            
            # Get first and last hidden state
            h_first = hidden_states[1] # Exclude [CLS]
            h_final = hidden_states[-2] # Exclude [SEP]
            
            # Get element-wise product and difference
            product = h_first * h_final
            difference = h_first - h_final
            
            # Concatenate
            span_representation = torch.cat([h_first, h_final, product, difference], dim=0)
            layer_representations.append(span_representation)
        
        representations.append({
            'representation': layer_representations,
            'label': label
        })
        
    return representations