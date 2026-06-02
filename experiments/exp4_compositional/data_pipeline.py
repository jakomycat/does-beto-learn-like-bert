from datasets import load_dataset

# Function to load SNLI dataset
def load_premises(split, max_len=20, max_samples=10):
    dataset = load_dataset('snli', split=split)
    
    dataset_premise = dataset['premise']
    dataset_premise = list(dict.fromkeys(dataset_premise)) # Only uniques
    
    # Filter by length
    dataset_filter = [sentence for sentence in dataset_premise if (len(sentence.split()) <= max_len)]
    
    # Trim the list (Only to test)
    if max_samples is not None:
        dataset_filter = dataset_filter[:max_samples]
    
    return dataset_filter