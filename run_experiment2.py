import torch
import argparse
import numpy as np

from src.extractor import load_model_and_tokenizer, get_cls_token
from experiments.exp2_probing_task.data_pipeline import load_probing_task
from experiments.exp2_probing_task.classifier import evaluate_all_layers

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument( # BERT or BETO
        '--lang',
        type=str,
        default='en'
    )
    
    parser.add_argument(
        '--task_name',
        type=str
    )
    
    args = parser.parse_args()
    lang, task_name = args.lang, args.task_name
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(lang=lang, device=device)
    
    # Get data
    data = load_probing_task(task_name)
    
    # Get all cls tokens
    X_train = get_cls_token(data['train']['sentences'], model, tokenizer, device, task_name, 'train')
    X_val = get_cls_token(data['val']['sentences'],   model, tokenizer, device, task_name, 'val')
    X_test = get_cls_token(data['test']['sentences'],  model, tokenizer, device, task_name, 'test')

    y_train = np.array(data['train']['labels'])
    y_val = np.array(data['val']['labels'])
    y_test = np.array(data['test']['labels'])
    
    # Train and evaluate
    num_classes = len(set(y_train))

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
    
if __name__ == '__main__':
    main()