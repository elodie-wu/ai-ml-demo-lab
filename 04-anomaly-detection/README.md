# Anomaly Detection

A comparison of Isolation Forest and Local Outlier Factor (LOF) for anomaly detection on a synthetic time-series dataset.

## Features

- Isolation Forest
- Local Outlier Factor (LOF)
- Feature preprocessing
- Hyperparameter comparison
- Recall and precision evaluation
- IF + LOF ensemble
- Anomaly visualization

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Run

From the repository root:

```bash
cd 04-anomaly-detection
jupyter notebook demo.ipynb
```

The notebook uses:

```text
GenSyn2122.csv
GenSyn_labels2122.csv
```

## Experiments

Two anomaly detection algorithms are compared:

- Isolation Forest
- Local Outlier Factor (LOF)

The experiments also explore different values of:

```text
Isolation Forest:
- contamination
- n_estimators

LOF:
- contamination
- n_neighbors
```

The models are evaluated using recall and precision.

An ensemble is also created by marking a sample as anomalous when either Isolation Forest or LOF detects it.

## Results

Isolation Forest tends to detect more anomalies but also produces more false positives.

LOF is more conservative and achieves better precision in the tested configurations.

The IF + LOF ensemble improves recall by combining the predictions of both models, but this also increases false positives.

This demonstrates the trade-off between recall and precision in anomaly detection.

## Project Structure

```text
04-anomaly-detection/
├── demo.ipynb
├── GenSyn2122.csv
├── GenSyn_labels2122.csv
└── README.md
```

## Notes

The dataset is highly imbalanced, with only a small number of anomalous samples.

For anomaly detection:

- Recall measures how many real anomalies are detected.
- Precision measures how many predicted anomalies are actually anomalies.
