import torch
import argparse
import numpy as np

from src.extractor import load_model_and_tokenizer, get_cls_token, set_seed
from experiments.exp2_probing_task.data_pipeline import load_probing_task
from experiments.exp2_probing_task.classifier import evaluate_all_layers

def get_data_for_task(task_name, model, tokenizer, device, lang):
    # Get data
    data = load_probing_task(task_name, lang)
    
    # Get all cls tokens
    X_train = get_cls_token(data['train']['sentences'], model, tokenizer, device, task_name, 'train', lang)
    X_val = get_cls_token(data['val']['sentences'],   model, tokenizer, device, task_name, 'val', lang)
    X_test = get_cls_token(data['test']['sentences'],  model, tokenizer, device, task_name, 'test', lang)

    y_train = np.array(data['train']['labels'])
    y_val = np.array(data['val']['labels'])
    y_test = np.array(data['test']['labels'])
    
    # Train and evaluate
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1

    if lang == 'en':
        model_name = 'bert'
    elif lang == 'es':
        model_name = 'beto'
    
    evaluate_all_layers(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        num_classes=num_classes,
        device=device,
        task_name=task_name,
        output_filename=f'probing_{model_name}'
    )

def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument( # BERT or BETO
        '--lang',
        type=str,
        default='en'
    )
    
    parser.add_argument(
        '--full_run',
        action=argparse.BooleanOptionalAction,
        default=True
    )
    
    parser.add_argument(
        '--task_name',
        type=str
    )
    
    args = parser.parse_args()
    lang, task_name, full_run = args.lang, args.task_name, args.full_run
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(lang=lang, device=device)
    
    if full_run:
        tasks = ['sentence_length', 'word_content', 'tree_depth', 'bigram_shift',
                 'past_present', 'subj_number', 'obj_number', 'odd_man_out', 'coordination_inversion']
        
        for task in tasks:
            get_data_for_task(task, model, tokenizer, device, lang)
            
    else:
        get_data_for_task(task_name, model, tokenizer, device, lang)
    
if __name__ == '__main__':
    main()