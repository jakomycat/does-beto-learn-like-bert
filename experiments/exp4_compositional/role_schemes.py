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