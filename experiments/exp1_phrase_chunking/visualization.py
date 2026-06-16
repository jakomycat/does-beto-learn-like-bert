import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE

LABEL_COLORS = {
    'NOUN':     '#fed481',
    'PROPN':    '#f98e52',
    'PRON':     '#e65c2e',
    'NUM':      '#b8d0c3',
    'VERB':     '#dd4a4c',
    'AUX':      '#9e0142',
    'ADJ':      '#86cfa5',
    'ADV':      '#3d95b8',
    'O':        '#5e4fa2',
}

def plot_tsne_layers(span_representations, layers, output_filename=None):
    labels = [s['label'] for s in span_representations] # Get labels
    unique_labels = list(set(labels))

    # Graph configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, layer in zip(axes, layers):
        vectors = np.array([s['representation'][layer] for s in span_representations])
        
        # Train t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=7, max_iter=1000)
        vectors_2d = tsne.fit_transform(vectors)
        
        # Plot each class separately to assign colors and legends
        for label in unique_labels:
            mask = [l == label for l in labels]
            ax.scatter(
                vectors_2d[mask, 0],
                vectors_2d[mask, 1],
                label=label,
                color=LABEL_COLORS.get(label),
                alpha=0.6,
                s=12
            )
        
        ax.set_title(f't-SNE - Layer {layer}')

        ax.set_xticks([]) 
        ax.set_yticks([])

    # Legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, bbox_to_anchor=(1.02, 0.5), loc='center left', markerscale=2)

    plt.suptitle('t-SNE 2D - Span Representations per layer')
    plt.tight_layout()
    
    # Create results/figures directory
    base = Path(__file__).resolve()
    route = base.parent.parent.parent / 'results' / 'figures'
    
    route.mkdir(parents=True, exist_ok=True)
    
    route_to_save = route / f'{output_filename}.png'
    
    if route_to_save.exists():
        print(f'The figure was overwritten at {output_filename}.png')
    else:
        print(f'The figure was successfully saved at {output_filename}.png')
    
    plt.savefig(route_to_save, dpi=300, bbox_inches='tight')