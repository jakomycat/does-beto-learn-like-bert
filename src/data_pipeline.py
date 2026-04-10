from datasets import load_dataset

import random

# Function to get raw data
def load_raw_dataset(lang='en'):
    # This is for BERT
    if lang == 'en':
        dataset = load_dataset('conll2000')
    
    # This is for BETO - Coming Soon
    elif lang == 'es':
        raise ValueError('Its not implemented')
        
    return dataset['train']

# IOB to chunks
def parse_iob_to_chunks(sentences):
    chunks = []
    
    labels = ['ADJP', 'ADVP', 'CONJP', 'INTJ', 'LST', 'NP', 'PP', 'PRT', 'QP', 'SBAR', 'VP']
    
    # Iterate in sentences
    for tokens, chunk_tags in zip(sentences['tokens'], sentences['chunk_tags']):
        
        chunk = {'text': '', 'label': ''} # Reset chunk
        
        for token, tag in zip(tokens, chunk_tags):
            
            # Case B-XX
            if tag % 2 == 1:
                if chunk['text'] != '': chunks.append(chunk) # Add found chunk
                
                label = labels[int(tag // 2)] # Get label
                chunk = {'text': token, 'label': label} # Initialize chunk
                
            # Case I-XX
            elif tag % 2 == 0 and tag != 0:
                chunk['text'] += ' ' + token
                
            # Case O
            elif tag == 0:
                if chunk['text'] != '':
                    chunks.append(chunk) # Add found chunk
                    chunk = {'text': '', 'label': ''} # Clear chunk
                    
        # Maybe a open chunk
        if chunk['text'] != '': chunks.append(chunk)
        
    return chunks            

# Auxiliar function for "extract_negative_spans"
def is_invalid_chunk(span_tags):
    # Case O is in tags
    if 0 in span_tags:
        return True
    
    first_B = None
    for span in span_tags:
        # Get first B-XX
        if first_B == None and span % 2 == 1:
            first_B = span
            
        # Case two B-XX
        if first_B != span and span % 2 == 1:
            return True
        
    # Doesn't exist B-XX
    if first_B == None:
        return True
    
    # Case when this begin with I-XX
    if first_B != span_tags[0]:
        return True
    
    # In other case this is valid chunk
    return False
    
# Function to extract no-chunks
def extract_negative_spans(sentences, n_needed=500, max_len=4, seed=7):
    negative_spans = []
    num_sentences = len(sentences['tokens'])
    
    random.seed(seed) 
    
    while len(negative_spans) < n_needed:
        chunk_text = ''
        
        # Get a random token
        s_idx = random.randint(0, num_sentences - 1)
        tokens = sentences['tokens'][s_idx]
        tags = sentences['chunk_tags'][s_idx]
        
        # Avoid one-word sentences
        if len(tokens) < 2:
            continue
        
        # Get a random split
        start_idx = random.randint(0, len(tokens) - 2)
        
        max_end = min(start_idx + max_len, len(tokens) - 1) # This is to prevent exceeding the limit
        end_idx = random.randint(start_idx + 1, max_end)
        
        chunk_text += " ".join(tokens[start_idx + 1:end_idx + 1]) # Get full text
        
        tags_to_validate = tags[start_idx:end_idx + 1]
        if is_invalid_chunk(tags_to_validate):
            negative_spans.append({
                'text': chunk_text,
                'label': 'None'
            })
        
    return negative_spans
    
# Function to mix the chunks with the non-chunks
def balance_and_sample(labeled_chunks, negative_spans):
    return
    
# Principal function
def get_phrasal_data(lang='en'):
    return