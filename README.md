# MovieLens Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval for
MovieLens 100K, with the same pipeline supporting MovieLens 1M. The project
covers chronological splitting, configurable negative sampling, full-catalog
evaluation, TensorBoard monitoring, controlled experiments, cold-start item
features, and FAISS retrieval.

## How the model works

```text
user_idx  -> UserTower -> user embedding ---+
                                               +-> similarity scores -> cross entropy
movie_idx -> ItemTower -> item embedding ----+
```

The default objective compares each observed user-movie pair with 64 uniformly
sampled unseen movies. The positive movie is placed in column zero, and cross
entropy trains it to outrank the sampled negatives. The training command also
supports in-batch, popularity-weighted, and hybrid observed-negative sampling.

## Reading order

Read the implementation in dependency order:

1. [`preprocess.py`](preprocess.py) loads MovieLens ratings, creates implicit
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
7. [`experiments/run_model_experiments.py`](experiments/run_model_experiments.py)
   compares embedding and similarity configurations.
8. [`experiments/run_negative_sampling_experiments.py`](experiments/run_negative_sampling_experiments.py)
   compares negative-sampling strategies.
9. [`experiments/run_multi_seed_experiment.py`](experiments/run_multi_seed_experiment.py)
   repeats the selected configuration and reports mean and standard deviation.
10. [`features/prepare_movie_features.py`](features/prepare_movie_features.py),
    [`models/feature_two_tower.py`](models/feature_two_tower.py), and
    [`experiments/run_cold_start_experiment.py`](experiments/run_cold_start_experiment.py)
    implement the content-feature cold-start experiment.
11. [`retrieval/build_index.py`](retrieval/build_index.py) and
    [`retrieval/retrieve.py`](retrieval/retrieve.py) build and query a FAISS
    item index.
12. [`tests/`](tests) verifies preprocessing, sampling, model behavior,
    evaluation metrics, monitoring, and retrieval.

```text
MovieLens ratings
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

## Quick start: MovieLens 100K

The repository defaults to MovieLens 100K paths. Rebuild the processed files,
train the selected configuration, and evaluate it with:

```bash
python preprocess.py
python -m training.train
python -m evaluation.evaluate \
  --checkpoint checkpoints/two_tower.pt \
  --processed-dir data/processed \
  --split val
```

Training defaults to 128-dimensional embeddings, 64 uniform negatives per
positive interaction, checkpoint selection by NDCG@10, and early stopping.
The training script uses CUDA when available, then Apple Metal (`mps`), and
otherwise CPU.

### TensorBoard

Training records loss, retrieval metrics, learning rate, throughput, epoch
duration, and embedding statistics. Keep training in one terminal and start the
dashboard in another:

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006`. Label runs with `--run-name`, or disable logging
with `--no-tensorboard`.

## MovieLens 1M

Download and extract MovieLens 1M so the ratings file is available at:

```text
data/raw/ml-1m/ratings.dat
```

Keep its artifacts separate from the default 100K pipeline:

```bash
python preprocess.py --dataset 1m

python -m training.train \
  --processed-dir data/processed-1m \
  --observed-negatives data/processed-1m/train_negatives.csv \
  --output checkpoints/two_tower_1m.pt \
  --run-name ml1m

python -m evaluation.evaluate \
  --processed-dir data/processed-1m \
  --checkpoint checkpoints/two_tower_1m.pt \
  --split val
```

Before a full run, verify the larger-data path with one epoch:

```bash
python -m training.train \
  --processed-dir data/processed-1m \
  --observed-negatives data/processed-1m/train_negatives.csv \
  --epochs 1 \
  --no-early-stopping \
  --output artifacts/ml1m_smoke.pt \
  --run-name ml1m_smoke
```

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

Place the MovieLens 100K metadata file at `data/raw/u.item`, then run:

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
cosine-similarity ranking. The exact index is appropriate for MovieLens and can
later be replaced behind the same interface by an approximate index for a much
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

## Committed benchmark results

The selected MovieLens 100K configuration was repeated with seeds 42, 43, and
44. These are validation results; the test split remains reserved.

| Method | Recall@10 | Recall@50 | Recall@100 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0626 | 0.2208 | 0.3429 | 0.0317 |
| Two-tower mean | 0.1157 | 0.3443 | 0.5134 | 0.0599 |

The two-tower standard deviations were 0.0032, 0.0006, 0.0128, and 0.0034,
respectively.

The committed one-epoch MovieLens 1M smoke run produced Recall@10/50/100 of
0.0434/0.1503/0.2459. Its popularity baseline produced
0.0457/0.1478/0.2408. This verifies the pipeline; it is not a tuned benchmark.

## Tests

```bash
python -m unittest discover -s tests
```
