import random

# Function to add left-to-right role
def left_to_right(words):
    num_words = len(words)
    roles = list(range(num_words))
    
    return roles

# Function to add right-to-left role
def right_to_left(words):
    num_words = len(words)
    roles = list(range(num_words - 1, -1, -1))
    
    return roles

# Function to add bag-of-words role
def bag_of_words(words):
    num_words = len(words)
    roles = [0] * num_words
    
    return roles

# Function to add bidirectional role
def bidirectional(words):
    left_right_role = left_to_right(words)
    right_left_role = right_to_left(words)
    
    roles = list(zip(left_right_role, right_left_role))
    
    return roles

# Function to add random-tree role
def random_tree_roles(words, seed=123):
    num_words = len(words)
    
    # Recursion
    if num_words == 0:
        return []
    if num_words == 1:
        return [''] 
    
    roles = ['']*num_words
    
    rng = random.Random(seed)
    
    # Auxiliar function to recursion
    def _build_tree(start, end, current_path):
        # Base case
        if start == end:
            roles[start] = current_path
            return
        
        # Recursive case
        split =  rng.randint(start, end - 1)
        
        _build_tree(start, split, current_path + 'L')
        _build_tree(split + 1, end, current_path + 'R')
        
    # It start the recursive function
    _build_tree(0, num_words - 1, '')
    
    return roles

# Function to add tree role
def tree_roles(consituency_tree, words):
    roles = []
    
    # Recursive function
    def _walk(node, current_path):
        # Base case
        if node.is_leaf():
            roles.append(current_path)
            return

        # Recursive case
        children = node.children
        k = len(children)
        
        if k == 1:
            _walk(children[0], current_path)
            
        elif k >= 2:
            for i, child in enumerate(children):
                if i == (k - 1):
                    next_path = current_path + ('R'*i)
                else:
                    next_path = current_path + ('R'*i) + 'L'
                    
                _walk(child, next_path)

    if consituency_tree:
        _walk(consituency_tree, '')
        
    # Wheel alignment check
    if len(roles) != len(words):
        raise ValueError(
         f'The parser generated {len(roles)} tokens, but your words list has {len(words)} elements. Check the tokenization before continuing.'
        )

    return roles

# Function to pre-parse tree roles using Stanza in batch
def preparse_tree_roles(sentences, nlp):
    # Process entire batch
    doc = nlp(sentences)
    
    if len(doc.sentences) != len(sentences):
        raise ValueError(
            f'Stanza mismatch. Expected {len(sentences)} sentences, but got {len(doc.sentences)}.'
        )
        
    corpus_roles = []
    for i, sent in enumerate(doc.sentences):
        words = sentences[i]
        tree = sent.constituency
        roles = tree_roles(tree, words)
        corpus_roles.append(roles)
        
    return corpus_roles

# Function to map roles into indices
def build_role_vocab(corpus_roles):
    # corpus_roles can be list of lists or a plane list
    if corpus_roles and isinstance(corpus_roles[0], list):
        flat_roles = [role for sentence_roles in corpus_roles for role in sentence_roles]
    else:
        flat_roles = corpus_roles
        
    unique_roles = set(flat_roles)
    sort_unique_roles = sorted(list(unique_roles)) # It's to reproducibility
    
    role_to_id = {role: idx for idx, role in enumerate(sort_unique_roles)}
    
    return role_to_id