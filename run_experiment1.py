import argparse
import torch

from experiments.exp1_phrase_chunking.data_pipeline import get_phrasal_data
from experiments.exp1_phrase_chunking.visualization import plot_tsne_layers
from experiments.exp1_phrase_chunking.evaluation import evaluate_kmeans_nmi

from src.extractor import load_model_and_tokenizer, get_span_representation, set_seed

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument( # BERT or BETO
        '--lang',
        type=str,
        default='en'
    )
    
    parser.add_argument( # Use original implementation from principal paper
        '--original',
        type=bool,
        default=True
    )
    
    parser.add_argument(
        '--n_chunks',
        type=int,
        default=3000
    )
    
    parser.add_argument(
        '--n_no_chunks',
        type=int,
        default=500
    )
    
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[1, 2, 11, 12]
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=7
    )
    
    args = parser.parse_args()
    
    lang, original, n_chunks, n_no_chunks, layers, seed = args.lang, args.original, args.n_chunks, args.n_no_chunks, args.layers, args.seed
    
    # Fix all randomness before anything else
    set_seed(seed)
    
    # Load modelo and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(lang=lang, device=device)
    
    # Get CoNLL-2000 data processed
    data = get_phrasal_data(lang=lang, n_chunks=n_chunks, n_no_chunks=n_no_chunks, use_original=original)
    
    # Get span representation
    representations = get_span_representation(span_samples=data, model=model, tokenizer=tokenizer, device=device)
    
    if lang == 'en':
        model_name = 'bert'
    elif lang == 'es':
        model_name = 'beto'
        
    if original:
        original_name = 'original'
    else:
        original_name = 'no_original'
        
    # Visualization    
    plot_tsne_layers(span_representations=representations, layers=layers, output_filename=f'tsne_{model_name}_{original_name}')

    # Evaluation with NMI
    evaluate_kmeans_nmi(span_representations=representations, output_filename=f'nmi_score_{model_name}_{original_name}')
    
    print('Experiment successfully completed')
    
if __name__ == '__main__':
    main()