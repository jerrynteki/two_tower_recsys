# MovieLens Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval on
MovieLens 100K. The repository covers chronological data splitting, contiguous
ID mappings, normalized user/item embeddings, and in-batch-negative training.

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
4. [`training/train.py`](training/train.py) connects the data and model, builds
   diagonal cross-entropy labels, optimizes the towers, and saves a checkpoint.
5. [`evaluation/evaluate.py`](evaluation/evaluate.py) scores the full movie
   catalog, filters training-seen items, and measures Top-K retrieval quality.
6. [`experiments/run_model_experiments.py`](experiments/run_model_experiments.py)
   compares representation and scoring choices under one controlled setup.
7. [`tests/`](tests) verifies embedding behavior, the training objective,
   seen-item filtering, and retrieval metrics.

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

Train the model:

```bash
python -m training.train --epochs 5
```

Evaluate full-catalog retrieval on the validation split:

```bash
python -m evaluation.evaluate --split val
```

Compare model configurations with the same seed, data, and optimizer:

```bash
python -m experiments.run_model_experiments --epochs 3
```

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

### Why chronological instead of random splitting?

Recommendation predicts future behavior from past behavior. A random split can
put a future interaction in training and an earlier interaction in evaluation,
creating temporal leakage and overly optimistic offline metrics.

### Why persist ID mappings?

An embedding row has meaning only under the mapping used during training. The
same mappings must be reused in validation and serving, so they are model
artifacts rather than temporary preprocessing details.
