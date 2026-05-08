import torch
import argparse

from src.extractor import load_model_and_tokenizer, extract_verb_features
from experiments.exp3_subject_verb.data_pipeline import run_full_pipeline, create_difficulty_buckets
from experiments.exp3_subject_verb.classifier import run_full_sva_evaluation

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
    
    # Get data
    data, _ = run_full_pipeline(tokenizer)
    
    buckets = create_difficulty_buckets(data['test'])
    
    # Get verb features
    X_train = extract_verb_features(data['train']['bert_input_ids'], data['train']['verb_token_index'], model, tokenizer, device, 'train')
    y_train = data['train']['label'].values
    
    X_val = extract_verb_features(data['valid']['bert_input_ids'], data['valid']['verb_token_index'], model, tokenizer, device, 'val')
    y_val = data['valid']['label'].values
    
    X_test = extract_verb_features(data['test']['bert_input_ids'], data['test']['verb_token_index'], model, tokenizer, device, 'test')
    y_test = data['test']['label'].values
    
    #
    run_full_sva_evaluation(
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val, 
        X_test=X_test, 
        y_test=y_test, 
        buckets=buckets, 
        num_classes=2, # Singular vs Plural
        device=device, 
        output_filename=f'sva_matrix_{args.lang}'
    )

if __name__ == '__main__':
    main()