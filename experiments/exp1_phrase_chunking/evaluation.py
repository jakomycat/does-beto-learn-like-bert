import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# Function to get NMI
def evaluate_kmeans_nmi(span_representations, output_filename=None, seed=7):
    true_labels = [s['label'] for s in span_representations] # Real labels
    num_layers = len(span_representations[0]['representation'])
    
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
        
    # Get directory to save this data
    base = Path(__file__).resolve()
    route = base.parent.parent.parent / 'results' / 'csv'
    
    route.mkdir(parents=True, exist_ok=True)
        
    # Organize data in pandas
    data = {
        'layer': range(len(nmi_scores)),
        'nmi_score': nmi_scores
    }
    
    df = pd.DataFrame(data)
    
    # Save
    route_to_save = route / f'{output_filename}.csv'
    
    if route_to_save.exists():
        print(f'The data was overwritten at {output_filename}.csv')
    else:
        print(f'The data was successfully saved at {output_filename}.csv')
        
    df.to_csv(route_to_save, index=False)
        
    return nmi_scores