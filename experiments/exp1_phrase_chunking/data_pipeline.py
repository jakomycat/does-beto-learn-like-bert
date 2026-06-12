from datasets import load_dataset

import random

# Function to get raw data
def load_raw_dataset(lang='en'):
    # This is for BERT
    if lang == 'en':
        dataset = load_dataset('universal_dependencies', 'en_ewt')
    
    # This is for BETO
    elif lang == 'es':
        dataset = load_dataset('universal_dependencies', 'es_ancora')
        
    return dataset['train']

# Function to build chunks from Universal Dependencies
def build_chunks_from_ud(tokens, upos, head, deprel, upos_names=None):
    """
    Builds flat chunks aligned with Anderson et al. (2019) and Lacroix (2018).
    Uses transitive climbing for absorption to mitigate non-projectivity loss.
    """
    # Allowed relations and valid nucleus tags
    ABSORB_DEPRELS = {'det', 'amod', 'nummod', 'flat', 'fixed', 'compound', 'aux', 'case', 'cop'}
    VALID_NUCLEUS_UPOS = {'NOUN', 'PROPN', 'PRON', 'VERB', 'AUX', 'ADJ', 'ADV', 'NUM'}
    
    sentence = " ".join(tokens)
    n = len(tokens)
    
    # Map UPOS safely
    mapped_upos = []
    for tag in upos:
        
        # Case integer tag
        if isinstance(tag, int):
            if upos_names and 0 <= tag < len(upos_names):
                mapped_upos.append(upos_names[tag])
            else:
                mapped_upos.append('X')
                
        # Case string tag
        else:
            mapped_upos.append(str(tag))
            
    # Nucleus assignment via transitive climbing
    token_to_nucleus = [None] * n
    
    # Valid UPOS is its own nucleus initially
    for i in range(n):
        if mapped_upos[i] in VALID_NUCLEUS_UPOS:
            token_to_nucleus[i] = i

    # Assign dependents climbing the tree
    for i in range(n):
        raw_h_init = head[i]
        
        # Skip MWT (Multi-Word Tokens)
        if isinstance(raw_h_init, str) and ('-' in raw_h_init or '.' in raw_h_init):
            continue
            
        curr = i
        visited = set()
        
        while True:
            if curr in visited: # Avoid cycles
                break
            visited.add(curr)
            
            raw_h = head[curr]
            
            # Check MWT during climb
            if isinstance(raw_h, str) and ('-' in raw_h or '.' in raw_h):
                break
                
            try:
                h = int(raw_h)
            except (ValueError, TypeError):
                break 
                
            # Root node, stop climbing
            if h == 0: 
                break
                
            h_idx = h - 1
            rel = deprel[curr]
            
            # Stop if relation is not absorbable
            if rel not in ABSORB_DEPRELS:
                break
                
            # Lacroix restriction: 'det' must precede the noun
            if rel == 'det' and curr > h_idx:
                break
                
            curr = h_idx # Climb to parent

        # Assign valid nucleus
        if 0 <= curr < n and mapped_upos[curr] in VALID_NUCLEUS_UPOS:
            token_to_nucleus[i] = curr

    # Group into contiguous runs
    runs = []
    if n > 0:
        current_nucleus = token_to_nucleus[0]
        current_start = 0
        for i in range(1, n):
            nuc = token_to_nucleus[i]
            if nuc != current_nucleus:
                if current_nucleus is not None:
                    runs.append((current_nucleus, current_start, i - 1)) # Add found run
                current_nucleus = nuc
                current_start = i
                
        # Add final open run
        if current_nucleus is not None:
            runs.append((current_nucleus, current_start, n - 1))
            
    # Strict contiguity policy
    token_to_chunk_span = [None] * n
    for nucleus, start, end in runs:
        if start <= nucleus <= end:
            for idx in range(start, end + 1):
                token_to_chunk_span[idx] = (start, end, nucleus)
                
    # Generate output chunks
    chunks_output = []
    i = 0
    while i < n:
        
        # Case valid chunk
        if token_to_chunk_span[i] is not None:
            start, end, nucleus = token_to_chunk_span[i]
            chunk_text = " ".join(tokens[start:end+1]) # Get full text
            
            label = mapped_upos[nucleus] # Get label
            
            chunks_output.append({
                'text': chunk_text,
                'label': label,
                'sentence': sentence,
                'start': start,
                'end': end
            }) # Add chunk
            
            i = end + 1 # Update index
            
        # Case no-chunk / 'O' label
        else:
            chunks_output.append({
                'text': tokens[i],
                'label': 'O',
                'sentence': sentence,
                'start': i,
                'end': i
            }) # Add no-chunk
            
            i += 1
            
    return chunks_output

# New fuunction to extract negative spans
def extract_negative_spans(sentences_tokens, gold_chunks_map, n_needed=500, max_len=4, seed=7):
    negative_spans = []
    seen_spans = set()
    num_sentences = len(sentences_tokens)
    
    rng = random.Random(seed)
    
    # It controls possible infinite loop
    max_attempts = n_needed * 20
    attempts = 0
    
    while len(negative_spans) < n_needed:
        if attempts >= max_attempts:
            raise RuntimeError(
                f'Attempt limit reached: I only found {len(negative_spans)} out of {n_needed} negative examples.'
                'Check the parameters (max_len).'
            )
        attempts += 1
        
        # Get a random sentence
        s_idx = rng.randint(0, num_sentences - 1)
        tokens = sentences_tokens[s_idx]
        
        if len(tokens) < 2:
            continue
        
        # We generate a span
        start_idx = rng.randint(0, len(tokens) - 2)
        max_end = min(start_idx + max_len - 1, len(tokens) - 1)
        end_idx = rng.randint(start_idx + 1, max_end)
        
        # It avoids exactly spans
        span_signature = (s_idx, start_idx, end_idx)
        if span_signature in seen_spans:
            continue
        
        # If it's a valid chunk, continue
        if (start_idx, end_idx) in gold_chunks_map.get(s_idx, set()):
            continue
        
        seen_spans.add(span_signature)
        span_tokens = tokens[start_idx:end_idx + 1]
        
        negative_spans.append({
            'text': ' '.join(span_tokens),
            'label': 'O',
            'sentence': ' '.join(tokens),
            'start': start_idx,
            'end': end_idx
        })
        
    return negative_spans
    
# Function to mix the chunks with the non-chunks
def balance_and_sample(labeled_chunks, negative_spans=None, n_chunks=3000, n_no_chunks=500, seed=7):
    """
    If negative_spans is None this indicates that it will use original paper's implementation
    """
    rng = random.Random(seed)
    
    # Original implementation
    if negative_spans is None:
        # Separate chunks by label
        non_o_chunks = [chunk for chunk in labeled_chunks if chunk['label'] != 'O']
        o_chunks = [chunk for chunk in labeled_chunks if chunk['label'] == 'O']
        
        # Sample from each group
        sample_non_o = rng.sample(non_o_chunks, n_chunks)
        sample_o = rng.sample(o_chunks, n_no_chunks)
        
        # Combine and shuffle
        final_list = sample_non_o + sample_o
    
    # Own extension
    else:
        # Get chunks
        non_o_chunks = [chunk for chunk in labeled_chunks if chunk['label'] != 'O']
        
        # Get a random sample of chunks
        sample_chunks = rng.sample(non_o_chunks, n_chunks)
        
        # Combine the lists
        final_list = sample_chunks + negative_spans
    
    return final_list
    
# Principal function
def get_phrasal_data(lang='en', n_chunks=3000, n_no_chunks=500, use_original=True, seed=7):
    # This is the overall process
    # Load dataset
    print('1/4 - Loading dataset')
    sentences = load_raw_dataset(lang)
    
    # Get valid chunks
    print('2/4 - Getting valid chunks and building gold map')
    labeled_chunks = []
    gold_chunks_map = {}
    upos_names = sentences.features['upos'].feature.names # Extract UPOS tags
    
    for s_idx, (tokens, upos, head, deprel) in enumerate(zip(sentences['tokens'], sentences['upos'], sentences['head'], sentences['deprel'])):
        # Extract chunks
        sentence_chunks = build_chunks_from_ud(tokens, upos, head, deprel, upos_names=upos_names)
        labeled_chunks.extend(sentence_chunks)
        
        valid_spans = set()
        for chunk in sentence_chunks:
            if chunk['label'] != 'O':
                valid_spans.add((chunk['start'], chunk['end']))
                
        gold_chunks_map[s_idx] = valid_spans
        

    # Skip negative span extraction
    if use_original:
        print('3/4 - Skipping negative span extraction (original paper mode)')
        negative_spans = None
    # Case extract invalid chunks (noise)
    else:
        print('3/4 - Getting invalid chunks (noise)')
        
        negative_spans = extract_negative_spans(
            sentences_tokens=sentences['tokens'],
            gold_chunks_map=gold_chunks_map,
            n_needed=n_no_chunks,
            seed=seed
        )
     
    print('4/4 - Balancing and mixing the final dataset')
    final_data = balance_and_sample(labeled_chunks, negative_spans, n_chunks=n_chunks, n_no_chunks=n_no_chunks, seed=seed)
    
    print(f'Pipeline completed, total samples: {len(final_data)}')
    
    return final_data