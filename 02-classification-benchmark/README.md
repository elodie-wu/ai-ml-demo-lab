# Classification Benchmark

A classification benchmark using the Breast Cancer Wisconsin dataset to compare preprocessing techniques, classification models, feature selection, dimensionality reduction, and cross-validation.

## Features

- Missing value imputation using mean and median
- Standardization and Min-Max normalization
- K-Nearest Neighbors (KNN)
- Decision Tree
- AdaBoost
- Random Forest
- Hyperparameter comparison
- Pearson correlation-based feature selection
- PCA dimensionality reduction
- 10-fold stratified cross-validation
- Evaluation using accuracy, precision, recall, and F1 score

## Requirements

- Python 3.10+
- NumPy
- Pandas
- scikit-learn

```bash
pip install numpy pandas scikit-learn
```

## Dataset

The project uses the Breast Cancer Wisconsin dataset for binary classification.

The target variable contains two classes:

- `M` — Malignant
- `B` — Benign

The labels are converted to:

```
M -> 1
B -> 0
```

The dataset is stored in:

```
data/breast-cancer.csv
```

## Run

From the repository root:

```bash
cd 02-classification-benchmark
python classification_benchmark.py
```

## Experiments

The script runs several classification experiments.

### 1. Data Preprocessing

Mean and median imputation are compared for handling missing values.

The effect of feature scaling is then evaluated using:

- No normalization
- StandardScaler
- MinMaxScaler

Both KNN and Decision Tree are tested to demonstrate how scaling affects different types of classifiers.

### 2. Hyperparameter Tuning

Several parameter values are compared for four classifiers:

| Classifier    | Hyperparameter | Values         |
| ------------- | -------------- | -------------- |
| KNN           | `n_neighbors`  | 3, 9, 15, 21   |
| Decision Tree | `max_depth`    | 2, 8, 14       |
| AdaBoost      | `n_estimators` | 10, 20, 30     |
| Random Forest | `n_estimators` | 10, 30, 50, 60 |

Each configuration is evaluated using accuracy, precision, recall, and F1 score.

### 3. Feature Selection

Pearson correlation is used to measure the relationship between each feature and the target.

Features with an absolute correlation greater than `0.6` are selected:

```
| correlation | > 0.6 |
```

Decision Tree performance before and after feature selection is then compared.

### 4. PCA Dimensionality Reduction

Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset.

Decision Trees with different maximum depths are evaluated before and after PCA.

The script also compares PCA with Pearson feature selection at a similar dimensionality.

### 5. Cross-Validation

10-fold stratified cross-validation is performed for:

- KNN (`k = 3`)
- KNN (`k = 9`)
- Decision Tree (`max_depth = 2`)
- Decision Tree (`max_depth = 8`)

For each model, the script reports:

- Accuracy for each fold
- F1 score for each fold
- Mean accuracy and standard deviation
- Mean F1 score and standard deviation

## Evaluation Metrics

The models are evaluated using:

| Metric    | Description                                                |
| --------- | ---------------------------------------------------------- |
| Accuracy  | Overall proportion of correct predictions                  |
| Precision | Proportion of positive predictions that are correct        |
| Recall    | Proportion of actual positive samples correctly identified |
| F1 Score  | Harmonic mean of precision and recall                      |

## Project Structure

```
02-classification-benchmark/
├── data/
│   └── breast-cancer.csv
├── classification_benchmark.py
└── README.md
```

## Notes

The dataset is split into 70% training data and 30% test data using stratified sampling.

Preprocessing parameters are fitted only on the training data and then applied to the test data to avoid data leakage.

A fixed random seed (`42`) is used where applicable to make the experiments reproducible.
