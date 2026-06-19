import argparse
import torch
import stanza

from src.extractor import load_model_and_tokenizer, get_cls_token, get_filler_embeddings
from experiments.exp4_compositional.data_pipeline import load_premises
from experiments.exp4_compositional.role_schemes import left_to_right, right_to_left, bag_of_words, bidirectional, build_role_vocab, preparse_tree_roles, random_tree_roles
from experiments.exp4_compositional.tpdn import map_roles_to_ids, run_tpdn_evaluation

def make_tree_scheme(roles_dict):
    def tree_scheme(words):
        return roles_dict[tuple(words)]
    return tree_scheme

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument( # BERT or BETO
        '--lang',
        type=str,
        default='en'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=10000
    )
    
    parser.add_argument(
        '--max_depth',
        type=int,
        default=4
    )
    
    args = parser.parse_args()
    lang, max_samples, max_depth = args.lang, args.max_samples, args.max_depth
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(lang=lang, device=device)
    
    # Get data
    sentences_train_raw = load_premises(split='train', max_samples=max_samples)
    sentences_test_raw  = load_premises(split='test', max_samples=max_samples)

    sentences_train = [s.split() for s in sentences_train_raw]
    sentences_test  = [s.split() for s in sentences_test_raw]
    
    # Initialize Stanza once
    nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos,constituency', tokenize_pretokenized=True)
    
    # Pre-parse syntax trees
    train_tree_roles = preparse_tree_roles(sentences_train, nlp, max_depth=max_depth)
    test_tree_roles  = preparse_tree_roles(sentences_test, nlp, max_depth=max_depth)
    
    # Build combined tree roles dictionary
    tree_roles_dict = {tuple(w): r for w, r in zip(sentences_train, train_tree_roles)}
    tree_roles_dict.update({tuple(w): r for w, r in zip(sentences_test, test_tree_roles)})
    
    # Get fillers and targets
    fillers_train = get_filler_embeddings(sentences_train, model, tokenizer, device)
    fillers_test  = get_filler_embeddings(sentences_test, model, tokenizer, device)
    
    targets_train = get_cls_token(sentences_train, model, tokenizer, device, task_name='tpdn', split='train', lang=lang, is_split_into_words=True)
    targets_test  = get_cls_token(sentences_test, model, tokenizer, device, task_name='tpdn', split='test', lang=lang, is_split_into_words=True)
    
    # Define role schemes
    role_schemes_dict = {
        'left_to_right': left_to_right,
        'right_to_left': right_to_left,
        'bag_of_words': bag_of_words,
        'bidirectional': bidirectional,
        'random_tree': lambda w: random_tree_roles(w, seed=123),
        'tree': make_tree_scheme(tree_roles_dict)
    }
    
    # Run probing
    run_tpdn_evaluation(
        fillers_train=fillers_train,
        fillers_test=fillers_test,
        targets_train=targets_train,
        targets_test=targets_test,
        sentences_train=sentences_train,
        sentences_test=sentences_test,
        role_schemes_dict=role_schemes_dict,
        build_vocab_fn=build_role_vocab,
        map_roles_fn=map_roles_to_ids,
        device=device,
        output_filename=f'tpdn_table4_{lang}',
        seeds=[42, 43, 44, 45, 46]
    )

if __name__ == '__main__':
    main()