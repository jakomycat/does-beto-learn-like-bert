import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.manifold import TSNE

def plot_tsne_layers(span_representations, layers, output_filename='tsne_span_representations.png'):
    labels = [s['label'] for s in span_representations] # Get labels
    unique_labels = list(set(labels))
    
    colors = plt.get_cmap("tab10")(np.arange(len(unique_labels))) # Palette color

    # Graph configuration
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, layer in zip(axes, layers):
        vectors = np.array([s['representation'][layer] for s in span_representations])
        
        # Train t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=7, max_iter=1000)
        vectors_2d = tsne.fit_transform(vectors)
        
        # Plot each class separately to assign colors and legends
        for label, color in zip(unique_labels, colors):
            mask = [l == label for l in labels]
            ax.scatter(
                vectors_2d[mask, 0],
                vectors_2d[mask, 1],
                label=label,
                color=color,
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'Figure saved at: {output_path}')