from datasets import load_dataset

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
def extract_negative_spans(sentences, n_needed=500):
    return
    
# Function to mix the chunks with the non-chunks
def balance_and_sample(labeled_chunks, negative_spans):
    return
    
# Principal function
def get_phrasal_data(lang='en'):
    return