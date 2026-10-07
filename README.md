# MovieLens Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval on
MovieLens 1M. The project uses one data protocol throughout: positive ratings
are filtered to a 5-core dataset, then each user's interactions are randomly
split into train, validation, and test partitions at 80/10/10.

## Project layout

```text
preprocess.py                     filter, map IDs, and create the one split
datasets.py                       load interaction rows for training
models/two_tower.py               user and movie towers
training/negative_sampling.py     sampled-softmax training objective
training/train.py                 training, validation, and checkpoints
evaluation/evaluate.py            full-catalog metrics and simple baselines
retrieval/                        embedding export and FAISS retrieval
features/                         movie metadata features
experiments/                       model, seed, and cold-start experiments
tests/                             preprocessing, model, and retrieval checks
```

The main recommendation path is:

```text
MovieLens ratings
    -> preprocess.py
    -> train/validation/test CSVs and ID mappings
    -> two-tower training
    -> full-catalog evaluation or FAISS retrieval
```

Each tower maps its user or movie ID to a normalized embedding. Training scores
the positive movie against 64 sampled unseen movies and uses cross-entropy to
rank the positive first.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract MovieLens 1M so `data/raw/ml-1m/ratings.dat` exists, then
prepare the dataset:

```bash
python preprocess.py
```

Preprocessing keeps ratings of 4 or 5, iteratively removes users and movies
with fewer than five positive interactions, and uses seed 2020 to shuffle the
interactions before splitting each user 80/10/10. The default output is
`data/processed-1m/`, containing the three partitions, zero-based ID
mappings, and protocol metadata. The MovieLens 1M reference counts are 6,034
users, 3,125 movies, and 574,376 positive interactions.

## Train

The model uses 128-dimensional embeddings, 64 uniformly sampled unseen movies
per positive, validation NDCG@10 for checkpoint selection, and early stopping
by default:

```bash
python -m training.train
```

Training writes the best checkpoint to `checkpoints/two_tower_1m.pt` and a
resumable checkpoint beside it. Training metrics and model statistics are
logged to TensorBoard under `runs/training/`:

```bash
tensorboard --logdir runs
```

Resume an interrupted run with:

```bash
python -m training.train \
  --resume checkpoints/two_tower_1m.latest.pt \
  --epochs 12
```

`--epochs` is the total epoch limit, including epochs already completed. Resume
restores the data path, model, sampling, optimizer, validation settings, and
random states from the latest checkpoint.

## Evaluate

Evaluate on the validation partition:

```bash
python -m evaluation.evaluate --split val --ks 10 20
```

Evaluation ranks the full movie catalog and masks items already present in the
user's training history. The test command also masks validation interactions:

```bash
python -m evaluation.evaluate --split test --ks 10 20
```

The evaluator reports multi-positive Recall@K and NDCG@K, plus random and
popularity reference rankings. Use `--no-baselines` to report only the trained
two-tower model.

## Experiments and retrieval

Repeat a selected configuration across seeds:

```bash
python -m experiments.run_multi_seed_experiment --seeds 42 43 44
```

Compare two-tower configurations:

```bash
python -m experiments.run_model_experiments --epochs 3
```

Prepare genre and release-year features and run the cold-start experiment:

```bash
python -m features.prepare_movie_features
python -m experiments.run_cold_start_experiment --epochs 5
```

Build an exact FAISS index and retrieve ten unseen movies for raw user ID 1:

```bash
python -m retrieval.build_index
python -m retrieval.retrieve --user-id 1 --top-k 10
```

`IndexFlatIP` performs exact inner-product search. Since both vectors are
L2-normalized, this is cosine similarity.

## Current reference result

The saved single-seed run selected epoch 10 by validation NDCG@10. Its held-out
test metrics on the 80/10/10 split were:

| Ranking | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0032 | 0.0036 | 0.0067 | 0.0047 |
| Popularity | 0.0905 | 0.0980 | 0.1457 | 0.1110 |
| Two-tower | 0.1604 | 0.1744 | 0.2421 | 0.1924 |

This is one seed; use the multi-seed experiment for a more stable estimate.
