from datasets import load_dataset, concatenate_datasets
import random

# Global cache to avoid reloading/parsing XNLI on every split
_ES_POOL_CACHE = None

def _build_es_pool(max_len, target_size=20000, seed=42, test_ratio=0.2):
    global _ES_POOL_CACHE
    if _ES_POOL_CACHE is not None:
        return _ES_POOL_CACHE

    # Function
    def _clean_unique(premises, seen, bucket, limit=None):
        for s in premises:
            if limit is not None and len(bucket) >= limit:
                break
            s = s.strip()
            if not s or len(s.split()) > max_len:
                continue
            if s in seen:
                continue
            seen.add(s)
            bucket.append(s)

    seen = set()
    pool = []

    # Load and process human translation
    val = load_dataset('facebook/xnli', 'es', split='validation')
    test = load_dataset('facebook/xnli', 'es', split='test')
    human = concatenate_datasets([val, test])
    
    _clean_unique(human['premise'], seen, pool)

    # Load and process machine translation, only if needed
    if len(pool) < target_size:
        train = load_dataset('facebook/xnli', 'es', split='train')
        _clean_unique(train['premise'], seen, pool, limit=target_size)

    pool = pool[:target_size]

    # Shuffle dataset reproducibly
    rng = random.Random(seed)
    rng.shuffle(pool)

    # Split dataset
    n_test = int(len(pool) * test_ratio)
    es_test = pool[:n_test]
    es_train = pool[n_test:]

    _ES_POOL_CACHE = {
        'train': es_train, 
        'test': es_test
    }
    
    return _ES_POOL_CACHE

# Function to load premises
def load_premises(split, lang='en', max_len=20, max_samples=10, target_size=20000):
    if lang == 'en':
        dataset = load_dataset('snli', split=split)

        dataset_premise = dataset['premise']
        dataset_premise = list(dict.fromkeys(dataset_premise))  # Only uniques

        # Filter by length
        dataset_filter = [s for s in dataset_premise if len(s.split()) <= max_len]
    elif lang == 'es':
        pool = _build_es_pool(max_len=max_len, target_size=target_size)
        dataset_filter = pool[split]

    # Trim the list (Only to test)
    if max_samples is not None:
        dataset_filter = dataset_filter[:max_samples]

    return dataset_filter