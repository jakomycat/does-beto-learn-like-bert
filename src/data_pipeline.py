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