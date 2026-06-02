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