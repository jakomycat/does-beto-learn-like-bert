import torch
import argparse
import numpy as np

from src.extractor import load_model_and_tokenizer, extract_verb_features

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument( # BERT or BETO
        '--lang',
        type=str,
        default='en'
    )
    
    args = parser.parse_args()
    lang = args.lang
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(lang=lang, device=device)

if __name__ == '__main__':
    main()