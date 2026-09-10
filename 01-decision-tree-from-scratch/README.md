# Decision Tree from Scratch

A decision tree for nominal features implemented from scratch with NumPy. Each feature value creates its own child branch instead of being forced into a binary split.

## Features

- Entropy and information gain
- Multiway splits for nominal features
- Recursive tree construction and stopping rules
- Prediction with fallback for unseen feature values
- Text-based tree export

## Requirements

- Python 3.10+
- NumPy

```bash
pip install numpy
```

## Run

From the repository root:

```bash
cd 01-decision-tree-from-scratch
python decision_tree.py data/rtg_A.csv assets/output_tree_A.txt
```

Replace `rtg_A.csv` with `rtg_B.csv` or `rtg_C.csv` to run the other datasets.

Optional parameters:

```bash
python decision_tree.py <input.csv> <output.txt> --max-depth 10 --min-samples 2
```

## Test

```bash
python -m unittest -v test_decision_tree.py
```

The tests cover multiway splitting, unseen-value fallback, and the minimum-sample stopping rule.

## Results

| Dataset | Training accuracy | Root feature | Root entropy | Root information gain | Root branches |
| ------- | ----------------- | ------------ | ------------ | --------------------- | ------------- |
| rtg_A   | 100%              | 0            | 0.4022       | 0.1167                | 2             |
| rtg_B   | 100%              | 3            | 0.9537       | 0.0532                | 2             |
| rtg_C   | 100%              | 3            | 0.9044       | 0.9044                | 50            |

`rtg_C` creates 50 branches at the root because feature 3 has about 50 distinct nominal values. This also demonstrates how high-cardinality features can lead to overfitting.

The reported accuracy is training accuracy. Full tree outputs are stored in `assets/`.
