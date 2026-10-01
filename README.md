# MovieLens Two-Tower Retrieval

A from-scratch PyTorch implementation of two-tower candidate retrieval on
MovieLens 1M. The repository covers chronological data splitting, contiguous
ID mappings, uniform negative sampling, side-information cold-start
experiments, full-catalog evaluation, and FAISS retrieval.

## Architecture

```text
user_idx -> UserTower -> user embedding ---+
                                             +-> similarity matrix -> cross entropy
movie_idx -> ItemTower -> item embedding ---+
```

The default objective compares each positive user-movie pair with 64 uniformly
sampled unseen movies. Cross entropy trains the positive movie, stored in column
zero, to outrank those negatives.

## Code walkthrough

Read the implementation in dependency order:

1. [`preprocess.py`](preprocess.py) loads MovieLens, keeps positive implicit
   feedback, maps raw IDs to contiguous indices, and creates chronological
   train, validation, and test splits.
2. [`datasets.py`](datasets.py) loads the processed user-item pairs and exposes
   them as PyTorch tensors for a `DataLoader`.
3. [`models/two_tower.py`](models/two_tower.py) defines the independent user
   and item encoders, normalized embeddings, and similarity scoring methods.
4. [`training/negative_sampling.py`](training/negative_sampling.py) implements
   uniform unseen-item sampling and the sampled-softmax training step.
5. [`training/train.py`](training/train.py) connects the data and model,
   selects the best validation checkpoint, logs TensorBoard monitoring data,
   and saves the result.
6. [`evaluation/evaluate.py`](evaluation/evaluate.py) scores the full catalog,
   filters training-seen items, selects Top-K candidates, calculates retrieval
   quality, and compares the model with random and popularity baselines.
7. [`experiments/run_model_experiments.py`](experiments/run_model_experiments.py)
   compares representation and scoring choices under one controlled setup.
8. [`experiments/run_multi_seed_experiment.py`](experiments/run_multi_seed_experiment.py)
    repeats a selected configuration and reports mean and standard deviation.
9. [`features/prepare_movie_features.py`](features/prepare_movie_features.py)
    turns genres and release year into an item-feature matrix.
10. [`models/feature_two_tower.py`](models/feature_two_tower.py) encodes movie
    content, including movies unseen during training.
11. [`experiments/run_cold_start_experiment.py`](experiments/run_cold_start_experiment.py)
    compares ID-only and content-based retrieval on held-out cold movies.
12. [`retrieval/build_index.py`](retrieval/build_index.py) and
    [`retrieval/retrieve.py`](retrieval/retrieve.py) build and query a FAISS index.
13. [`tests/`](tests) verifies preprocessing, sampling, model behavior,
    evaluation metrics, monitoring, features, and retrieval.

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
evaluation/evaluate.py -> full-catalog retrieval metrics
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

Download and extract MovieLens 1M so these files exist:

```text
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Prepare the ratings:

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
train/validation/total epoch duration, and embedding norms. Start the
TensorBoard dashboard in another terminal:

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006`. Use `--run-name` to label a run or
`--no-tensorboard` to disable logging:

```bash
python -m training.train --epochs 5 --run-name dim128 --embedding-dim 128
```

Training also atomically saves a resumable checkpoint after every completed
epoch. By default, `checkpoints/two_tower_1m.latest.pt` contains the latest model,
Adam optimizer, completed epoch, best model and validation results, early-stopping
counter, and Python, NumPy, PyTorch CPU/CUDA/MPS, shuffle, and negative-sampling
random states. The `--output` file remains the best-model export for evaluation
and retrieval. Use `--latest-checkpoint` to choose a different resume-file path.

Resume an interrupted run with:

```bash
python -m training.train --resume checkpoints/two_tower_1m.latest.pt --epochs 12
```

`--epochs` is the total epoch limit, including epochs already completed. Resume
restores the saved data path, model, sampling, optimizer, and validation settings;
corresponding CLI settings are ignored. Output paths and TensorBoard options
come from the new command, and logging starts a new run at the resumed epoch.
For a custom output location, pass `--output` again. A checkpoint that has already
early-stopped stays stopped unless `--no-early-stopping` is supplied.

Resume starts at the next epoch; work from an interrupted partial epoch is
repeated. Keep the processed data, device, software environment, and total epoch
limit unchanged to reproduce the original trajectory (subject to backend
determinism). Extending the epoch limit is supported, but the trainer also
evaluates at the final epoch, so changing that limit can change validation timing.
Older best-model exports do not contain enough state to resume training.

Evaluate full-catalog retrieval on the validation split. This reports the
two-tower model and fixed random and popularity baselines under the same
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

Prepare the MovieLens 1M genres and release years, then run the content
cold-start comparison:

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

## FlowCF-compatible benchmark

The primary pipeline uses a chronological leave-two-out split because it more
closely resembles future recommendation. A separate benchmark protocol matches
the [FlowCF authors' RecBole configuration](https://github.com/chengkai-liu/FlowCF):
ratings of at least 4 are positive, users and movies must have at least five
positive interactions, seed 2020 randomly orders the interactions before an
80/10/10 split within each user, and evaluation ranks the full catalog with
multi-positive Recall and NDCG at 10 and 20. Preprocessing validates the
reference result of 6,034 users, 3,125 real movies, and 574,376 interactions.
The paper/RecBole count is 3,126 items because RecBole includes its reserved
padding ID; this project has no padding item.

Create the separate benchmark artifacts without replacing the chronological
data:

```bash
python preprocess.py --protocol flowcf
```

Train the same two-tower architecture on that split:

```bash
python -m training.train \
  --processed-dir data/processed-1m-flowcf \
  --output checkpoints/two_tower_1m_flowcf_protocol.pt \
  --ks 10 20 \
  --selection-metric NDCG@10 \
  --patience 10 \
  --run-name flowcf_protocol
```

Evaluate the selected checkpoint once on the held-out test split:

```bash
python -m evaluation.evaluate \
  --processed-dir data/processed-1m-flowcf \
  --checkpoint checkpoints/two_tower_1m_flowcf_protocol.pt \
  --split test \
  --ks 10 20
```

For test evaluation under this protocol, both training and validation
interactions are masked. These results can be compared with FlowCF only when
FlowCF is run on the same exported split; published runtime numbers are not
comparable across different hardware.

The selected epoch-10 checkpoint produced the following held-out test results
on the exported FlowCF-compatible split:

| Method | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0032 | 0.0036 | 0.0067 | 0.0047 |
| Popularity | 0.0905 | 0.0980 | 0.1457 | 0.1110 |
| Two-tower | 0.1604 | 0.1744 | 0.2421 | 0.1924 |

## MovieLens 1M data pipeline

Ratings of 4 or 5 are treated as positive implicit feedback. Raw user and movie
IDs are mapped to contiguous embedding indices. For every user with at least
three positive interactions, the latest event is held out for test, the
previous event for validation, and all earlier events for training.

Ratings below 4 are excluded from the implicit-positive training set. For each
positive pair, training samples 64 movies that the user has not previously
liked and teaches the model to rank the positive movie above them. All commands
use `data/processed-1m` by default; custom paths remain available through
explicit CLI options.

Start with a one-epoch smoke run before launching the full training command:

```bash
python -m training.train --epochs 1 --no-early-stopping \
  --output artifacts/ml1m_smoke.pt --run-name ml1m_smoke
```

The verified one-epoch smoke run produced Recall@10/50/100 of approximately
0.043/0.150/0.247. Use the full early-stopped run for model selection rather
than treating the smoke result as a benchmark.

## Current validation result

The current full MovieLens 1M baseline uses seed 42 and selects epoch 9 by
validation NDCG@10. It exceeds the popularity baseline at every reported K:

| Method | Recall@10 | Recall@50 | Recall@100 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0457 | 0.1478 | 0.2408 | 0.0218 |
| Two-tower | 0.0772 | 0.2547 | 0.3886 | 0.0370 |

This is a single-seed validation result. Repeat the selected configuration with
multiple seeds before final reporting; the test split remains reserved until
model selection is complete.

### Why chronological instead of random splitting?

Recommendation predicts future behavior from past behavior. A random split can
put a future interaction in training and an earlier interaction in evaluation,
creating temporal leakage and overly optimistic offline metrics.

### Why persist ID mappings?

An embedding row has meaning only under the mapping used during training. The
same mappings must be reused in validation and serving, so they are model
artifacts rather than temporary preprocessing details.
