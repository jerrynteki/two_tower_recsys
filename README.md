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

