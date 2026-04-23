import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# Function to get NMI
def evaluate_kmeans_nmi(span_representations, num_layers=12, output_filename=None, seed=7):
    true_labels = [s['label'] for s in span_representations] # Real labels
    
    # Get n_clusters - This is how many labels there are
    unique_labels = list(set(true_labels))
    n_clusters = len(unique_labels)
    
    nmi_scores = []

    for layer in range(num_layers):
        # Get span repr
        X = np.array([s['representation'][layer] for s in span_representations])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Calculate NMI score
        nmi = normalized_mutual_info_score(true_labels, clusters)
        nmi_scores.append(nmi)
        
    # Save this data in csv
    data = {
        'layer': range(len(nmi_scores)),
        'nmi_score': nmi_scores
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(output_filename, index=False)
        
    return nmi_scores