# Clustering Comparison

A comparison of K-means and DBSCAN clustering on a two-dimensional dataset. The experiments explore how different parameter settings affect cluster structure and noise detection.

## Features

- K-means clustering
- DBSCAN density-based clustering
- K-means comparison with different values of `k`
- DBSCAN parameter tuning with `eps` and `min_samples`
- Noise detection with DBSCAN
- Two-dimensional cluster visualization

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

```bash
pip install numpy matplotlib scikit-learn jupyter
```

## Run

From the repository root:

```bash
cd 03-clustering-comparison
jupyter notebook demo.ipynb
```

Run the notebook cells in order.

The dataset is loaded from:

```python
X = np.load("part1_data.npz")["X"]
```

## K-means

K-means was tested with:

```text
k = 2
k = 3
k = 4
k = 5
```

The final model uses:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)
```

`k = 3` provides the best representation of the visible cluster structure.

With `k = 2`, distinct groups are merged together. With `k = 4` or `k = 5`, existing clusters are split into smaller groups.

K-means captures the overall structure well, although the irregular shape of the central cluster causes some boundary mixing.

## DBSCAN

Two DBSCAN configurations are used to demonstrate different density settings.

### Three clusters with minimal noise

```text
eps = 1.5
min_samples = 20
```

Result:

```text
Clusters: 3
Noise points: 3
```

This configuration identifies the three main dense regions while leaving only a small number of sparse points as noise.

### Densest cluster cores

```text
eps = 0.5
min_samples = 18
```

Result:

```text
Clusters: 3
Noise points: 207
```

The smaller neighbourhood radius keeps the densest regions of each cluster and classifies more low-density points as noise.

## Results

| Method  | Parameters                      | Clusters | Noise |
| ------- | ------------------------------- | -------: | ----: |
| K-means | `k = 3`                         |        3 |   N/A |
| DBSCAN  | `eps = 1.5`, `min_samples = 20` |        3 |     3 |
| DBSCAN  | `eps = 0.5`, `min_samples = 18` |        3 |   207 |

K-means assigns every point to a cluster and requires the number of clusters to be specified in advance.

DBSCAN detects clusters based on density and can explicitly identify low-density points as noise.

For this dataset, both methods identify three main groups, but DBSCAN provides more control over how sparse regions are treated.

## Project Structure

```text
03-clustering-comparison/
├── assets/
│   ├── k-means2.png
│   ├── k-means3.png
│   ├── k-means4.png
│   ├── k-means5.png
│   ├── dbscan-eps1.5-min20.png
│   └── dbscan-eps0.5-min18.png
├── demo.ipynb
├── part1_data.npz
└── README.md
```

## Notes

The dataset contains 1,400 samples with two numerical features.

K-means uses `random_state=42` to make the clustering reproducible.

In DBSCAN, label `-1` represents noise, while labels `0`, `1`, `2`, and so on represent detected clusters.
