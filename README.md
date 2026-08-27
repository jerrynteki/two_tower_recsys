# MovieLens Two-Tower Retrieval

A seven-day learning project that grows from chronological interaction data to
a PyTorch two-tower model, retrieval evaluation, negative-sampling experiments,
side features, and ANN serving.

## Day 1: data pipeline

The first pipeline treats MovieLens ratings of 4 or 5 as positive implicit
feedback. Raw user and movie IDs are mapped to contiguous embedding indices.
For every user with at least three positive interactions, the latest event is
held out for test, the previous event for validation, and all earlier events
for training.

Run from the project root:

```bash
source .venv/bin/activate
python preprocess.py
```

The reusable outputs are written to `data/processed/`.

### Why chronological instead of random splitting?

Recommendation predicts future behavior from past behavior. A random split can
put a future interaction in training and an earlier interaction in evaluation,
creating temporal leakage and overly optimistic offline metrics.

### Why persist ID mappings?

An embedding row has meaning only under the mapping used during training. The
same mappings must be reused in validation and serving, so they are model
artifacts rather than temporary preprocessing details.

