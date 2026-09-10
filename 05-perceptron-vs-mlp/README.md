# Perceptron vs MLP

A comparison of a Perceptron implemented from scratch and a Multi-Layer Perceptron (MLP) on synthetic classification datasets.

## Features

- Perceptron implemented from scratch with NumPy
- Perceptron training with different numbers of epochs
- Comparison on SeaSyn and RingSyn datasets
- Multi-Layer Perceptron using scikit-learn
- Feature standardization
- Sigmoid vs ReLU activation comparison
- Classification accuracy evaluation

## Requirements

- Python 3.10+
- NumPy
- Pandas
- scikit-learn

```bash
pip install numpy pandas scikit-learn
```

## Run

From the repository root:

```bash
cd 05-perceptron-vs-mlp
```

Run the Perceptron experiments:

```bash
python perceptron.py
```

Run the MLP:

```bash
python mlp.py
```

Compare Sigmoid and ReLU activation functions:

```bash
python activation_comparison.py
```

## Experiments

The Perceptron is trained on both SeaSyn and RingSyn with different numbers of epochs.

The MLP is trained on the RingSyn dataset using:

```text
Hidden layers: 1
Hidden neurons: 8
Activation: ReLU
Optimizer: Adam
Max iterations: 1000
```

A separate experiment compares Sigmoid and ReLU activation functions using the same MLP architecture.

## Results

The experiments demonstrate the difference between a linear Perceptron and a neural network with a hidden layer.

The Perceptron performs well when the classes can be separated by a linear decision boundary, while the MLP can model more complex non-linear relationships.

The activation comparison also shows how the choice of activation function can affect training behaviour and classification performance.

## Project Structure

```text
05-perceptron-vs-mlp/
├── data/
│   ├── RingSynTest.csv
│   ├── RingSynTrain.csv
│   ├── SeaSynTest.csv
│   └── SeaSynTrain.csv
├── activation_comparison.py
├── mlp.py
├── perceptron.py
└── README.md
```

## Notes

The Perceptron uses a learning rate of `0.1` and is implemented without scikit-learn.

The MLP uses standardized features and a fixed `random_state=42` for reproducibility.
