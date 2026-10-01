# MovieLens 1M Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval for
MovieLens 1M. The project covers chronological splitting, configurable negative
sampling, full-catalog evaluation, TensorBoard monitoring, controlled
experiments, cold-start item features, and FAISS retrieval.

## How the model works

```text
user_idx  -> UserTower -> user embedding ---+
                                               +-> similarity scores -> cross entropy
movie_idx -> ItemTower -> item embedding ----+
```

The default objective compares each observed user-movie pair with 64 uniformly
sampled unseen movies. The positive movie is placed in column zero, and cross
entropy trains it to outrank the sampled negatives. Training also supports
in-batch, popularity-weighted, and hybrid observed-negative sampling.

## Reading order

1. [`preprocess.py`](preprocess.py) loads MovieLens 1M ratings, creates implicit
   feedback, maps raw IDs to embedding indices, and writes chronological train,
   validation, and test splits.
2. [`datasets.py`](datasets.py) exposes processed user-movie pairs to a PyTorch
   `DataLoader`.
3. [`models/two_tower.py`](models/two_tower.py) defines the independent user and
   item towers and their similarity scoring.
4. [`training/negative_sampling.py`](training/negative_sampling.py) implements
   catalog sampling and sampled-softmax training.
5. [`training/train.py`](training/train.py) trains the model, evaluates each
   epoch, performs early stopping, logs TensorBoard data, and saves the best
   checkpoint.
6. [`evaluation/evaluate.py`](evaluation/evaluate.py) performs full-catalog
   retrieval, masks training-seen movies, reports Recall, HitRate, MRR, and
   NDCG, and compares against random and popularity baselines.
7. [`experiments/`](experiments) contains controlled model, negative-sampling,
   multi-seed, and cold-start experiments.
8. [`features/prepare_movie_features.py`](features/prepare_movie_features.py)
   converts `movies.dat` genres and release years into model features.
9. [`retrieval/build_index.py`](retrieval/build_index.py) and
   [`retrieval/retrieve.py`](retrieval/retrieve.py) build and query a FAISS item
   index.
10. [`tests/`](tests) verifies preprocessing, sampling, model behavior,
    evaluation metrics, monitoring, features, and retrieval.

```text
MovieLens 1M ratings
       |
       v
preprocess.py -> CSV splits and ID mappings
       |
       v
datasets.py -> DataLoader batches
       |
       v
models/two_tower.py -> user and item embeddings
       |
       v
training/train.py -> best checkpoint + TensorBoard logs
       |
       +-> evaluation/evaluate.py -> full-catalog metrics and baselines
       |
       +-> retrieval/build_index.py -> FAISS index -> recommendations
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and extract MovieLens 1M so these files exist:

```text
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

## Run the pipeline

All defaults point to MovieLens 1M:

```bash
python preprocess.py
python -m training.train
python -m evaluation.evaluate --split val
```

Training defaults to 128-dimensional embeddings, 64 uniform negatives per
positive interaction, checkpoint selection by NDCG@10, and early stopping. It
uses CUDA when available, then Apple Metal (`mps`), and otherwise CPU. The best
checkpoint is saved to `checkpoints/two_tower_1m.pt`.

Before a full run, verify the pipeline with one epoch:

```bash
python -m training.train \
  --epochs 1 \
  --no-early-stopping \
  --output artifacts/ml1m_smoke.pt \
  --run-name ml1m_smoke
```

## TensorBoard

Training records loss, retrieval metrics, learning rate, throughput, epoch
duration, and embedding statistics. Keep training in one terminal and start the
dashboard in another:

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006`. Label runs with `--run-name`, or disable logging
with `--no-tensorboard`.

## Experiments

Compare model configurations under the same data, seed, and optimizer:

```bash
python -m experiments.run_model_experiments --epochs 3
```

Compare in-batch, uniform, popularity-weighted, and hard negatives:

```bash
python -m experiments.run_negative_sampling_experiments --epochs 3
```

Repeat the selected training configuration across seeds and save trial-level
and aggregate results:

```bash
python -m experiments.run_multi_seed_experiment --seeds 42 43 44
```

### Negative-sampling options

- `uniform` is the validated default and samples unseen movies uniformly.
- `popularity` samples unseen movies in proportion to training frequency.
- `hybrid` mixes unseen movies with explicit ratings of 1 or 2 observed before
  the validation boundary.
- `in_batch` treats other positive movies in the batch as negatives.

For hybrid sampling:

```bash
python -m training.train --negative-strategy hybrid
```

### Cold-start item features

Create genre and release-year features from MovieLens 1M `movies.dat`, then run
the cold-start comparison:

```bash
python -m features.prepare_movie_features
python -m experiments.run_cold_start_experiment --epochs 5
```

## Retrieval index

Build an exact FAISS inner-product index and retrieve ten unseen movies for raw
user ID 1:

```bash
python -m retrieval.build_index
python -m retrieval.retrieve --user-id 1 --top-k 10
```

Because the vectors are L2-normalized, inner-product ranking is equivalent to
cosine-similarity ranking. The exact index is appropriate for MovieLens 1M and
can later be replaced behind the same interface by an approximate index for a
larger catalog.

## Data protocol

Ratings of 4 or 5 are positive implicit feedback. For every eligible user, the
latest positive is test, the previous positive is validation, and earlier
positives are training data. This chronological split avoids training on an
interaction that happened after the event being evaluated.

Ratings of 1 or 2 before the validation boundary are exported to
`train_negatives.csv` for optional hybrid sampling. Raw user and movie IDs are
persisted as contiguous mappings because the same embedding-row meanings must
be reused during validation and retrieval.

## Current verified result

The committed one-epoch MovieLens 1M smoke run produced Recall@10/50/100 of
0.0434/0.1503/0.2459. Its popularity baseline produced
0.0457/0.1478/0.2408. This verifies the end-to-end pipeline; it is not a tuned
benchmark.

## Tests

```bash
python -m unittest discover -s tests
```
