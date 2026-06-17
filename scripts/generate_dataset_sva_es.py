import ast
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Function to parse UD morphological features into a dict
def parse_ud_feats(feats_str):
    if isinstance(feats_str, dict):
        return feats_str
    if not feats_str or feats_str == '_':
        return {}

    s = str(feats_str).strip()

    # Try to parse a dict-like string
    if s.startswith('{'):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return {}

    # Parse standard UD format (key=value|key=value)
    feats_dict = {}
    for item in s.split('|'):
        if '=' in item:
            k, v = item.split('=', 1)
            feats_dict[k] = v
    return feats_dict

# Function to safely convert a head value to int
def safe_head_to_int(head):
    h = str(head)
    if h.lstrip('-').isdigit():
        return int(h)
    return None

# Function to detect relative pronouns
def is_relative_pronoun(token, upos):
    relative_pronouns = {'que', 'quien', 'quienes', 'cual', 'cuales'}
    return upos == 'PRON' and token.lower() in relative_pronouns

# Function to check if a candidate is dominated by the subject in the tree
def is_dominated_by_subject(cand_idx, subject_idx, heads_int):
    seen = set()
    current = cand_idx
    while True:
        if current in seen:
            return False
        seen.add(current)

        head = heads_int[current]
        if head is None or head == 0:
            return False
        head_idx = head - 1
        if head_idx == subject_idx:
            return True
        if head_idx < 0 or head_idx >= len(heads_int):
            return False
        current = head_idx

# Function to count intervening nouns and attractors between subject and verb
def count_intervening_stats(upos_list, feats_list, start_idx, end_idx, subj_number,
                            subject_idx, heads_int,
                            strict_attractors=True, include_pron=False):
    attractor_pos = {'NOUN', 'PROPN'}
    if include_pron:
        attractor_pos.add('PRON')

    n_intervening = 0
    n_diff_intervening = 0

    for k in range(start_idx, end_idx):
        # Skip tokens that aren't potential attractors
        if upos_list[k] not in attractor_pos:
            continue
        if strict_attractors and not is_dominated_by_subject(k, subject_idx, heads_int):
            continue

        token_feats = parse_ud_feats(feats_list[k])
        token_number = token_feats.get('Number')

        # Count attractors and those differing from the subject number
        if token_number:
            n_intervening += 1
            if token_number != subj_number:
                n_diff_intervening += 1

    return n_intervening, n_diff_intervening

# Function to extract subject-verb agreement pairs from a single example
def extract_sva_pairs(example, upos_feature, split_name,
                      strict_attractors=True, include_pron=False):
    # Filter by length
    tokens = example['tokens']
    if len(tokens) > 50:
        return []

    upos = [upos_feature.int2str(t) for t in example['upos']]
    deprels = example['deprel']
    heads_int = [safe_head_to_int(h) for h in example['head']]
    feats = example['feats']

    pairs = []

    for i, pos in enumerate(upos):
        if pos != 'VERB':
            continue

        # Get verb features
        verb_idx = i
        verb_feats = parse_ud_feats(feats[verb_idx])
        verb_number = verb_feats.get('Number')
        verb_form = verb_feats.get('VerbForm')
        verb_person = verb_feats.get('Person')

        if not verb_number or verb_form != 'Fin' or not verb_person:
            continue

        # Find the nominal subject of the verb
        subject_idx = -1
        for j, (head_i, deprel) in enumerate(zip(heads_int, deprels)):
            if head_i == (verb_idx + 1) and deprel == 'nsubj':
                subject_idx = j
                break

        if subject_idx == -1 or is_relative_pronoun(tokens[subject_idx], upos[subject_idx]):
            continue

        # Get subject features
        subj_feats = parse_ud_feats(feats[subject_idx])
        subj_number = subj_feats.get('Number')

        if not subj_number or subj_number != verb_number:
            continue

        # Define the span between subject and verb
        start_idx = min(subject_idx, verb_idx) + 1
        end_idx = max(subject_idx, verb_idx)

        n_intervening, n_diff_intervening = count_intervening_stats(
            upos, feats, start_idx, end_idx, subj_number,
            subject_idx, heads_int,
            strict_attractors=strict_attractors, include_pron=include_pron
        )

        # Save the extracted pair
        pairs.append({
            'sentence': ' '.join(tokens),
            'context': ' '.join(tokens[:verb_idx + 1]),
            'subject': tokens[subject_idx],
            'verb': tokens[verb_idx],
            'verb_pos': verb_idx,
            'subject_number': subj_number,
            'verb_number': verb_number,
            'verb_form': verb_form,
            'verb_person': verb_person,
            'n_intervening': n_intervening,
            'n_diff_intervening': n_diff_intervening,
            'distance': abs(subject_idx - verb_idx),
            'split_origin': split_name
        })

    return pairs

# Function to build the full SVA dataset from the AnCora corpus
def extract_ancora_sva_dataset(strict_attractors=True, include_pron=False):
    print('Loading es_ancora corpus...')
    dataset = load_dataset('universal_dependencies', 'es_ancora')
    extracted_data = []

    for split_name in dataset.keys():
        print(f'Processing split: {split_name}')
        data_split = dataset[split_name]
        upos_feature = data_split.features['upos'].feature

        for example in data_split:
            pairs = extract_sva_pairs(
                example, upos_feature, split_name,
                strict_attractors=strict_attractors, include_pron=include_pron
            )
            extracted_data.extend(pairs)

    print(f'Extracted {len(extracted_data)} valid SVA pairs.')
    return pd.DataFrame(extracted_data)

# Function to save the dataset in CSV and JSONL formats
def save_dataset(df, output_dir, project_root):
    output_csv = output_dir / 'dataset_sva.csv'
    output_jsonl = output_dir / 'dataset_sva.jsonl'

    df.to_csv(output_csv, index=False, encoding='utf-8')
    df.to_json(output_jsonl, orient='records', lines=True, force_ascii=False)

    print(f'Saved CSV: {output_csv.relative_to(project_root)}')
    print(f'Saved JSONL: {output_jsonl.relative_to(project_root)}')

def run_ancora_sva_pipeline(strict_attractors=True, include_pron=False):
    # Get route
    base = Path(__file__).resolve()
    project_root = base.parent.parent
    output_dir = project_root / 'generated_data' / 'ancora_sva_es'

    output_dir.mkdir(parents=True, exist_ok=True)

    df = extract_ancora_sva_dataset(strict_attractors=strict_attractors, include_pron=include_pron)
    save_dataset(df, output_dir, project_root)

    print('Pipeline completed successfully.')

if __name__ == '__main__':
    run_ancora_sva_pipeline()