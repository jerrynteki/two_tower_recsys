# MovieLens Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval on
MovieLens 100K. The repository covers chronological data splitting, contiguous
ID mappings, multiple negative-sampling strategies, side-information cold-start
experiments, full-catalog evaluation, and FAISS retrieval.

## Architecture

```text
user_idx -> UserTower -> user embedding ---+
                                             +-> similarity matrix -> cross entropy
movie_idx -> ItemTower -> item embedding ---+
```

For a batch of `B` positive user-item pairs, the model produces a `B x B`
similarity matrix. Entry `(i, i)` is the observed positive pair; other items in
the batch act as negatives for user `i`. Cross entropy therefore uses
`[0, 1, ..., B-1]` as its labels.

## Code walkthrough

Read the implementation in dependency order:

1. [`preprocess.py`](preprocess.py) loads MovieLens, keeps positive implicit
   feedback, maps raw IDs to contiguous indices, and creates chronological
   train, validation, and test splits.
2. [`datasets.py`](datasets.py) loads the processed user-item pairs and exposes
   them as PyTorch tensors for a `DataLoader`.
3. [`models/two_tower.py`](models/two_tower.py) defines the independent user
   and item encoders, normalized embeddings, and in-batch similarity matrix.
4. [`training/train.py`](training/train.py) connects the data and model, trains
   with configurable negatives, selects the best validation checkpoint, logs
   TensorBoard monitoring data, and saves the result.
5. [`evaluation/evaluate.py`](evaluation/evaluate.py) scores the full catalog,
   filters training-seen items, selects Top-K candidates, calculates retrieval
   quality, and compares the model with random and popularity baselines.
6. [`experiments/run_model_experiments.py`](experiments/run_model_experiments.py)
   compares representation and scoring choices under one controlled setup.
7. [`tests/`](tests) verifies embedding behavior, the training objective,
   seen-item filtering, and retrieval metrics.
8. [`training/negative_sampling.py`](training/negative_sampling.py) implements
   uniform, popularity-weighted, and model-selected hard negatives.
9. [`experiments/run_negative_sampling_experiments.py`](experiments/run_negative_sampling_experiments.py)
   compares those samplers with the in-batch baseline.
10. [`experiments/run_multi_seed_experiment.py`](experiments/run_multi_seed_experiment.py)
    repeats a selected configuration and reports mean and standard deviation.
11. [`features/prepare_movie_features.py`](features/prepare_movie_features.py)
    turns genres and release year into an item-feature matrix.
12. [`models/feature_two_tower.py`](models/feature_two_tower.py) encodes movie
    content, including movies unseen during training.
13. [`experiments/run_cold_start_experiment.py`](experiments/run_cold_start_experiment.py)
    compares ID-only and content-based retrieval on held-out cold movies.
14. [`retrieval/build_index.py`](retrieval/build_index.py) and
    [`retrieval/retrieve.py`](retrieval/retrieve.py) build and query a FAISS index.

```text
MovieLens interactions
        |
        v
preprocess.py -> processed CSV files
        |
        v
datasets.py -> DataLoader batches
        |
        v
models/two_tower.py -> user and item embeddings
        |
        v
training/train.py -> trained checkpoint
        |
        v
evaluation/evaluate.py -> Recall@K and HitRate@K
        |
        v
experiments/run_model_experiments.py -> comparison table
```

When reading each file, ask: What does it receive? What transformation does it
perform? What does it return or save? Which later component consumes that
output?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

Prepare the data:

```bash
python preprocess.py
```

Train the selected model. The defaults are the validated configuration:
128-dimensional embeddings, 64 uniform negatives per positive, best-checkpoint
selection by NDCG@10, and early stopping.

```bash
python -m training.train
```

Training logs loss, Recall, HitRate, MRR, NDCG, learning rate, throughput,
epoch duration, and embedding norms. Start the TensorBoard dashboard in another
terminal:

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006`. Use `--run-name` to label a run or
`--no-tensorboard` to disable logging:

```bash
python -m training.train --epochs 5 --run-name dim128 --embedding-dim 128
```

Evaluate full-catalog retrieval on the validation split. This reports the
two-tower model and permanent random and popularity baselines under the same
seen-item masking rules:

```bash
python -m evaluation.evaluate --split val
```

Repeat the selected configuration across three seeds and save both trial-level
and mean/std summaries:

```bash
python -m experiments.run_multi_seed_experiment --seeds 42 43 44
```

Compare model configurations with the same seed, data, and optimizer:

```bash
python -m experiments.run_model_experiments --epochs 3
```

Each configuration appears as a separate TensorBoard run, allowing its loss,
validation Recall@K, speed, and embedding statistics to be compared directly.

Compare negative-sampling strategies:

```bash
python -m experiments.run_negative_sampling_experiments --epochs 3
```

For the content experiment, place the MovieLens 100K `u.item` file at
`data/raw/u.item`, then run:

```bash
python -m features.prepare_movie_features
python -m experiments.run_cold_start_experiment --epochs 5
```

Build an exact FAISS index and retrieve ten unseen movies for raw user ID 1:

```bash
python -m retrieval.build_index
python -m retrieval.retrieve --user-id 1 --top-k 10
```

`IndexFlatIP` performs exact inner-product search. Because both sides are L2
normalized before indexing and querying, the score is cosine similarity. This
is simple and exact for MovieLens; the same interface can later use an
approximate FAISS index for a much larger catalog.

Run the tests:

```bash
python -m unittest discover -s tests
```

The training script automatically uses Apple Metal (`mps`) when available and
otherwise falls back to CPU. It saves the model weights and configuration under
`checkpoints/`.

## Data pipeline

Ratings of 4 or 5 are treated as positive implicit feedback. Raw user and movie
IDs are mapped to contiguous embedding indices. For every user with at least
three positive interactions, the latest event is held out for test, the
previous event for validation, and all earlier events for training.

Ratings of 1 or 2 that occurred before the validation period are preserved in
`train_negatives.csv`. They can be mixed with sampled unseen movies by running:

```bash
python -m training.train --negative-strategy hybrid
```

The validated default remains uniform sampling because it performed better in
the current controlled runs.

### MovieLens 1M

The same preprocessing and training interfaces support MovieLens 1M. Extract
the official archive so that the ratings file is at
`data/raw/ml-1m/ratings.dat`, then keep its outputs separate from 100K:

```bash
python preprocess.py --dataset 1m
python -m training.train --processed-dir data/processed-1m \
  --observed-negatives data/processed-1m/train_negatives.csv
python -m evaluation.evaluate --processed-dir data/processed-1m --split val
```

Start with a one-epoch smoke run before launching the full training command:

```bash
python -m training.train --processed-dir data/processed-1m \
  --observed-negatives data/processed-1m/train_negatives.csv \
  --epochs 1 --no-early-stopping --output artifacts/ml1m_smoke.pt
```

The verified one-epoch smoke run produced Recall@10/50/100 of
0.0434/0.1503/0.2459. The corresponding popularity values were
0.0457/0.1478/0.2408. This confirms the larger-data path works, but it is not a
tuned 1M benchmark; model selection should use a longer multi-seed run.

## Current validation result

The selected configuration was repeated with seeds 42, 43, and 44. The mean
full-catalog results all exceed the popularity baseline:

| Method | Recall@10 | Recall@50 | Recall@100 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0626 | 0.2208 | 0.3429 | 0.0317 |
| Two-tower mean | 0.1157 | 0.3443 | 0.5134 | 0.0599 |

The standard deviations were 0.0032, 0.0006, 0.0128, and 0.0034 respectively.
These are validation results; the test split remains reserved for final
reporting.

### Why chronological instead of random splitting?

Recommendation predicts future behavior from past behavior. A random split can
put a future interaction in training and an earlier interaction in evaluation,
creating temporal leakage and overly optimistic offline metrics.

### Why persist ID mappings?

An embedding row has meaning only under the mapping used during training. The
same mappings must be reused in validation and serving, so they are model
artifacts rather than temporary preprocessing details.
