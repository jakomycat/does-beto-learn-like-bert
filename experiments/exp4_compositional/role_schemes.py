import random
from tqdm import tqdm

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
def random_tree_roles(words, seed=123, max_depth=None):
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
            roles[start] = current_path[:max_depth] if max_depth is not None else current_path
            return
        
        # Recursive case
        split =  rng.randint(start, end - 1)
        
        _build_tree(start, split, current_path + 'L')
        _build_tree(split + 1, end, current_path + 'R')
        
    # It start the recursive function
    _build_tree(0, num_words - 1, '')
    
    return roles

# Function to add tree role
def tree_roles(consituency_tree, words, max_depth=None):
    roles = []

    def _walk_children(children, current_path):
        # Binarize a list of sibling subtrees, right-branching.
        k = len(children)
        if k == 1:
            _walk(children[0], current_path)
        else:
            # first child -> L ; rest -> R (recursively binarized)
            _walk(children[0], current_path + 'L')
            _walk_children(children[1:], current_path + 'R')

    def _walk(node, current_path):
        if node.is_leaf():
            final_path = current_path[:max_depth] if max_depth is not None else current_path
            roles.append(final_path)
            return

        children = node.children
        if len(children) == 1:
            # unary node: inherit path, no branching
            _walk(children[0], current_path)
        else:
            _walk_children(children, current_path)

    if consituency_tree:
        _walk(consituency_tree, '')

    # Wheel alignment check
    if len(roles) != len(words):
        raise ValueError(
         f'The parser generated {len(roles)} tokens, but your words list has {len(words)} elements. Check the tokenization before continuing.'
        )

    return roles

# Function to pre-parse tree roles using Stanza in batch
def preparse_tree_roles(sentences, nlp, max_depth=None, chunk_size=50):
    corpus_roles = []

    # Process in small chunks
    for start in tqdm(range(0, len(sentences), chunk_size), desc='Parsing trees (Stanza)'):
        chunk = sentences[start:start + chunk_size]
        doc = nlp(chunk)

        if len(doc.sentences) != len(chunk):
            raise ValueError(
                f'Stanza mismatch in chunk starting at {start}. '
                f'Expected {len(chunk)} sentences, but got {len(doc.sentences)}.'
            )

        for i, sent in enumerate(doc.sentences):
            words = chunk[i]
            tree = sent.constituency
            roles = tree_roles(tree, words, max_depth=max_depth)
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

    # Reserve id 0 for <unk>
    role_to_id = {'<unk>': 0}
    for idx, role in enumerate(sort_unique_roles, start=1):
        role_to_id[role] = idx

    return role_to_id