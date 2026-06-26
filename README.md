# Does BETO learn like BERT?

This project sets out to analyze how BERT and BETO models internally learn linguistic structures, and what differences may be caused by the morphosyntactic and typological complexity of Spanish compared to English. Following the methodology proposed by Jawahar et al. (2019), the four experiments are reproduced: phrasal syntax, probing tasks, subject-verb agreement and compositional structure.

The results and discussion can be found in the paper [coming soon].

## Reproducibility

To produce the results, Python version 3.11.9 and a GTX 1650 GPU were used. The experiments work on any Python 3.11.x version.

To install, clone the repository and set up the virtual environment:

```bash
git clone https://github.com/jakomycat/does-beto-learn-like-bert.git
cd does-beto-learn-like-bert
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

Also, the seeds used to run the experiments are the ones fixed by default; in certain experiments you can easily change the seed to see differences between runs. For more information see the `How to run the experiments` section.

## About the models

The pre-trained models available on HuggingFace are used:

- **English (BERT):** `bert-base-uncased` (Devlin et al., 2019).
- **Spanish (BETO):** `dccuchile/bert-base-spanish-wwm-uncased` (Cañete et al., 2020).

Both are uncased models with 12 layers, 768 hidden size dimensions and 12 attention heads. In all experiments, unless otherwise indicated, the activation of the `[CLS]` token at each layer is used as the sentence representation, following Jawahar et al. (2019).

## About the datasets

Each experiment automatically downloads or generates the data it needs. Below is a detail of which dataset each experiment uses:

| Experiment | English (BERT) | Spanish (BETO) |
|-------------|---------------|----------------|
| Phrase chunking | Universal Dependencies, `en_ewt` | Universal Dependencies, `es_ancora` |
| Probing tasks | X-Probe / SentEval (en) | X-Probe (es) |
| Subject-verb agreement | Linzen et al. (2016) dataset | Generated from UD `es_ancora` (see script) |
| Compositional structure | SNLI (premises) | XNLI (premises, es) |

**Sources and implementation notes:**

- **Phrase chunking (Exp. 1) - difference from the original paper.** Unlike Jawahar et al. (2019), who use the CoNLL-2000 chunking dataset (Tjong Kim Sang & Buchholz, 2000), here the chunks do not come from CoNLL-2000. They are built from the Universal Dependencies treebanks (`en_ewt` and `es_ancora`, loaded via HuggingFace `datasets`; Nivre et al., 2020) through a custom algorithm for deriving flat chunks by *transitive climbing* over the dependency tree, inspired by Lacroix (2018) and Anderson and Gómez-Rodríguez (2019). This decision was made to have chunks that are comparable and consistent between English and Spanish from the same source (UD), since there is no direct equivalent of CoNLL-2000 for Spanish. The detail of the algorithm is in `experiments/exp1_phrase_chunking/data_pipeline.py`.

- **X-Probe / SentEval (Exp. 2):** the probing tasks are downloaded from the [`ltgoslo/xprobe`](https://github.com/ltgoslo/xprobe) repository. The original tasks were proposed by Conneau et al. (2018) and extended to the multilingual domain (including Spanish) by Ravishankar et al. (2019). **Note:** 9 of the 10 tasks from the original paper are reproduced; `top_constituents` (TopConst) is omitted because the multilingual version of X-Probe does not include it in any language.

- **Subject-verb agreement (Exp. 3):** Following the formulation of the task as number prediction (Linzen et al., 2016) and BERT/BETO's masked LM objective, the verb token is replaced by [MASK] before extracting its representation. The binary classifier (singular/plural) therefore operates over the representation of the masked position at each layer, measuring to what extent the model infers the number of the verb from the context (subject and attractors) without observing the verb directly.
  
  - *English:* the subject-verb number dataset from Linzen et al. (2016) is used. The file (`agr_50_mostcommon_10K.tsv.gz`) is downloaded automatically from the authors' public link (hosted on Dropbox). If that link were to become unavailable, the dataset can also be obtained from the original Linzen et al. repository ([`TalLinzen/rnn_agreement`](https://github.com/TalLinzen/rnn_agreement)); in that case it is enough to place the `.tsv.gz` in `data/SVADataset/`.
  
  - *Spanish:* an equivalent dataset is generated from the UD `es_ancora` treebank (the Universal Dependencies version of the AnCora corpus by Taulé et al., 2008), using `scripts/generate_dataset_sva_es.py`. The already-generated dataset is included in `generated_data/ancora_sva_es/` so it can be reproduced without running the generation again.

- **SNLI / XNLI (Exp. 4):** the compositional structure analysis is reproduced through Tensor Product Decomposition Networks (TPDN), a method proposed by McCoy et al. (2019) and adopted by Jawahar et al. (2019) to probe whether sentence representations encode compositional structure. The premises from SNLI (Bowman et al., 2015) are used for English and from XNLI (Conneau et al., 2018b) for Spanish, both loaded via HuggingFace `datasets`. For Spanish, a pool of unique premises is built by combining the `validation` and `test` splits of XNLI (human translation) and, if the target size needs to be completed, part of the `train` split; the pool is shuffled reproducibly and divided into train/test. For English, the unique premises from SNLI filtered by length are taken. In both cases, `--max_samples` trims the number of premises after filtering.

## Repository structure

```
does-beto-learn-like-bert/
├── run_experiment1.py          # Launcher Exp. 1 — Phrase chunking (per-layer phrase representation)
├── run_experiment2.py          # Launcher Exp. 2 — Probing tasks (9 SentEval/xprobe tasks over the [CLS] token)
├── run_experiment3.py          # Launcher Exp. 3 — Subject-verb agreement (SVA) by difficulty buckets
├── run_experiment4.py          # Launcher Exp. 4 — Compositional structure (TPDN, role schemes)
│
├── src/
│   └── extractor.py            # Shared core: model/tokenizer loading (BERT/BETO),
│                               # seed fixing, and representation extraction
│                               # (spans, [CLS] token, verb features, embedding matrix)
│
├── experiments/                # Logic specific to each experiment
│   ├── exp1_phrase_chunking/
│   │   ├── data_pipeline.py    # Builds flat chunks from Universal Dependencies (en_ewt / es_ancora)
│   │   ├── evaluation.py       # K-means clustering + NMI per layer
│   │   └── visualization.py    # t-SNE projection of the per-layer representations
│   ├── exp2_probing_task/
│   │   ├── data_pipeline.py    # Downloads the probing tasks from the xprobe repo (ltgoslo/xprobe)
│   │   └── classifier.py       # Probing classifier and layer-by-layer evaluation
│   ├── exp3_subject_verb/
│   │   ├── data_pipeline.py    # Loads SVA data (EN: Linzen dataset; ES: UD es_ancora, see scripts/)
│   │   └── classifier.py       # Binary classifier (singular/plural) and evaluation by buckets
│   └── exp4_compositional/
│       ├── data_pipeline.py    # Premises from SNLI (en) and XNLI-es (es)
│       ├── role_schemes.py     # Role schemes: left-to-right, right-to-left, bag-of-words,
│       │                       # bidirectional, random-tree and tree (parsing with Stanza)
│       └── tpdn.py             # Tensor Product Decomposition Network: training and evaluation
│
├── scripts/
│   └── generate_dataset_sva_es.py   # Generates the Spanish SVA dataset from UD es_ancora
│
├── generated_data/
│   └── ancora_sva_es/
│       ├── dataset_sva.csv      # Already-generated SVA-es dataset (included to reproduce without re-running)
│       └── dataset_sva.jsonl    # Same data in JSONL format
│
├── results/
│   ├── csv/                     # Numerical results (one CSV per experiment × model)
│   │   ├── nmi_score_{bert,beto}_{original,no_original}.csv   # Exp. 1
│   │   ├── probing_{bert,beto}.csv                            # Exp. 2
│   │   ├── sva_matrix_{en,es}.csv                             # Exp. 3
│   │   └── tpdn_table4_{en,es}.csv                            # Exp. 4
│   └── figures/                 # Generated figures
│       └── tsne_{bert,beto}_{original,no_original}.png        # Exp. 1 (t-SNE)
│
├── requirements.txt            # Dependencies with pinned versions (PyTorch + CUDA 12.1)
├── LICENSE                     # MIT License
└── README.md
```

## How to run the experiments

All experiments are launched from the project root. Each one runs a single model at a time, selected with `--lang` (`en` for BERT, `es` for BETO), and writes its results to `results/csv/` and figures to `results/figures/` when applicable.

To reproduce the full set of results you must run each experiment twice, once for BETO and once for BERT.

### Experiment 1 - Phrase chunking

```bash
python run_experiment1.py --lang en # BERT
python run_experiment1.py --lang es # BETO
```

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `en` | `en` (BERT) or `es` (BETO) |
| `--original` / `--no-original` | `--original` | Jawahar et al. (2019)-style configuration, where the negative spans are chunks labeled as `O` sampled from the treebank itself, or the proposed variant (`--no-original`), which generates synthetic negative spans |
| `--n_chunks` | `3000` | Number of spans with a chunk to sample |
| `--n_no_chunks` | `500` | Number of `O`/negative spans. In `--original` mode these are spans labeled as `O` from the treebank; in `--no-original` mode these are synthetically generated negative spans |
| `--layers` | `1 2 11 12` | Layers to visualize with t-SNE |
| `--seed` | `7` | Seed |

**Outputs:** `results/csv/nmi_score_{bert,beto}_{original,no_original}.csv`, `results/figures/tsne_{bert,beto}_{original,no_original}.png`.

To also generate the variant proposed in this work:

```bash
python run_experiment1.py --lang en --no-original # BERT
python run_experiment1.py --lang es --no-original # BETO
```

### Experiment 2 — Probing tasks

```bash
# Runs the 9 tasks (by default)
python run_experiment2.py --lang en # BERT
python run_experiment2.py --lang es # BETO
```

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `en` | `en` (BERT) or `es` (BETO) |
| `--full_run` / `--no-full_run` | `--full_run` | Runs the 9 tasks, or a single one with `--task_name` |
| `--task_name` | — | Individual task (requires `--no-full_run`) |

To run a single task:

```bash
python run_experiment2.py --lang en --no-full_run --task_name tree_depth
```

Valid tasks (9): `sentence_length`, `word_content`, `tree_depth`, `bigram_shift`,
`past_present`, `subj_number`, `obj_number`, `odd_man_out`, `coordination_inversion`.

**Outputs:** `results/csv/probing_{bert,beto}.csv`.

### Experiment 3 — Subject-verb agreement (SVA)

```bash
python run_experiment3.py --lang en # BERT
python run_experiment3.py --lang es # BETO
```

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `en` | `en` (BERT) or `es` (BETO) |

**Outputs:** `results/csv/sva_matrix_{en,es}.csv`.

### Experiment 4 — Compositional structure (TPDN)

```bash
# By default it runs both languages (en and es) in a single execution
python run_experiment4.py
```

**Available flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--langs` | `en es` | Languages to process (one or several) |
| `--max_samples` | `10000` | Maximum number of premises per split (trimming applied after filtering by length) |
| `--max_depth` | `None` | Maximum depth of the role tree (`None` = no limit) |
| `--seeds` | `42 43 44 45 46` | Seeds to average over (multi-seed) |

For a single language:

```bash
python run_experiment4.py --langs en
```

**Outputs:** `results/csv/tpdn_table4_{en,es}.csv`.

## References

Anderson, M., Gómez-Rodríguez, C. (2019). Artificially Evolved Chunks for Morphosyntactic
Analysis. In *Proceedings of the 18th International Workshop on Treebanks and Linguistic
Theories (TLT, SyntaxFest 2019)* (pp. 133–143). ACL.
https://aclanthology.org/W19-7815/

Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus
for learning natural language inference. In *Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing* (pp. 632–642). ACL.
https://aclanthology.org/D15-1075/

Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish
Pre-trained BERT Model and Evaluation Data. In *PML4DC at ICLR 2020*.
https://github.com/dccuchile/beto

Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). What you
can cram into a single vector: Probing sentence embeddings for linguistic properties.
In *Proceedings of the 56th Annual Meeting of the Association for Computational
Linguistics* (pp. 2126–2136). ACL. https://aclanthology.org/P18-1198/

Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov,
V. (2018b). XNLI: Evaluating Cross-lingual Sentence Representations. In *Proceedings of
the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 2475–2485).
ACL. https://aclanthology.org/D18-1269/

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies* (pp. 4171–4186). ACL.
https://aclanthology.org/N19-1423/

Jawahar, G., Sagot, B., & Seddah, D. (2019). What Does BERT Learn about the
Structure of Language? In *Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics* (pp. 3651–3657). ACL.
https://aclanthology.org/P19-1356/

Lacroix, O. (2018). Investigating NP-Chunking with Universal Dependencies for English.
In *Proceedings of the Second Workshop on Universal Dependencies (UDW 2018)* (pp. 85–90).
ACL. https://aclanthology.org/W18-6010/

Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the Ability of LSTMs to Learn
Syntax-Sensitive Dependencies. *Transactions of the Association for Computational
Linguistics*, 4, 521–535. https://aclanthology.org/Q16-1037/

McCoy, R. T., Linzen, T., Dunbar, E., & Smolensky, P. (2019). RNNs Implicitly Implement
Tensor Product Representations. In *International Conference on Learning Representations
(ICLR 2019)*. https://openreview.net/forum?id=BJx0sjC5FX

Nivre, J., de Marneffe, M.-C., Ginter, F., Hajič, J., Manning, C. D., Pyysalo, S.,
Schuster, S., Tyers, F., & Zeman, D. (2020). Universal Dependencies v2: An Evergrowing
Multilingual Treebank Collection. In *Proceedings of the 12th Language Resources and
Evaluation Conference (LREC 2020)* (pp. 4034–4043). ELRA.
https://aclanthology.org/2020.lrec-1.497/

Ravishankar, V., Øvrelid, L., & Velldal, E. (2019). Probing Multilingual Sentence
Representations With X-Probe. In *Proceedings of the 4th Workshop on Representation
Learning for NLP (RepL4NLP-2019)* (pp. 156–168). ACL.
https://aclanthology.org/W19-4318/

Taulé, M., Martí, M. A., & Recasens, M. (2008). AnCora: Multilevel Annotated Corpora for
Catalan and Spanish. In *Proceedings of the Sixth International Conference on Language
Resources and Evaluation (LREC 2008)*. ELRA.
http://www.lrec-conf.org/proceedings/lrec2008/

Tjong Kim Sang, E. F., & Buchholz, S. (2000). Introduction to the CoNLL-2000 Shared Task:
Chunking. In *Proceedings of CoNLL-2000 and LLL-2000* (pp. 127–132). ACL.
https://aclanthology.org/W00-0726/