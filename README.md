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

The trainer supports `mlp`, `embedding`, and `residual_mlp` ID-only models, plus
`feature_hybrid` and `gated_feature_hybrid` item towers. Both combine a movie-ID
embedding with projected genre/year features; the gated version learns the
relative weight of the two vectors. Prepare movie features first:

```bash
python -m features.prepare_movie_features
python -m training.train \
  --item-tower gated_feature_hybrid \
  --user-tower embedding
```

`--item-tower` selects the item tower and `--user-tower` selects the user tower
used with `feature_hybrid`. The user tower accepts
`embedding`, `mlp`, or `residual_mlp`. Both flags accept their previous names
(`--architecture` and `--user-architecture`) as aliases.

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

## LightGCN

LightGCN represents the training interactions as a user-movie bipartite graph.
It repeatedly averages neighboring embeddings across graph layers, then scores
user-movie pairs with an inner product. Unlike the two-tower models above, it
uses BPR pairwise ranking loss and learns collaborative signals through graph
propagation. The graph is built from training interactions only.

Train a 64-dimensional, three-layer LightGCN with BPR:

```bash
python -m training.train_lightgcn \
  --processed-dir data/processed-1m \
  --output checkpoints/lightgcn_ml1m_d64_l3_lr0p001_reg1e-4_b4096_s42.pt \
  --epochs 100 \
  --batch-size 4096 \
  --embedding-dim 64 \
  --layers 3 \
  --learning-rate 0.001 \
  --regularization 0.0001 \
  --patience 10 \
  --seed 42
```

The trainer selects the checkpoint by validation NDCG@10. Evaluate its test
metrics after training:

```bash
python -m evaluation.evaluate_lightgcn \
  --checkpoint checkpoints/lightgcn_ml1m_d64_l3_lr0p001_reg1e-4_b4096_s42.pt \
  --processed-dir data/processed-1m \
  --split test \
  --ks 10 20
```

## Experiments and retrieval

Repeat a selected configuration across seeds:

```bash
python -m experiments.run_multi_seed_experiment --seeds 42 43 44
```

The same runner accepts `--architecture feature_hybrid` after movie features
have been prepared.

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

## Latest completed test results

These runs use the same held-out test split from `data/processed-1m`; test
evaluation masks both training and validation interactions. Each result is a
single seed, so treat it as an initial comparison rather than a multi-seed
estimate.

| Model (seed 42) | Best epoch | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 | Test Recall@20 | Test NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gated feature-hybrid (128d, embedding user tower) | 45 | 0.2049 | 0.2202 | 0.2364 | 0.3194 | 0.2567 |
| LightGCN (64d, 3 layers) | 100 | 0.1546 | 0.1623 | 0.1812 | 0.2448 | 0.1986 |

The gated feature-hybrid run used learning rate 0.1, temperature 0.15,
64 negatives, and batch size 256. LightGCN used learning rate 0.001,
regularization 0.0001, and batch size 4096. LightGCN's best validation epoch
was the 100-epoch limit, so it may need a longer run before treating it as a
fully tuned baseline.
