# AI/ML Demo Lab

A collection of small, self-contained implementations and experiments covering core Artificial Intelligence, Machine Learning, and Evolutionary Computation concepts.

The repository focuses on understanding how algorithms work through practical implementations, comparisons, and experiments rather than only using high-level APIs.

## Projects

| #   | Project                                                               | Topics                                                              |
| --- | --------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 01  | [Decision Tree from Scratch](./01-decision-tree-from-scratch)         | Entropy, information gain, multiway splits                          |
| 02  | [Classification Benchmark](./02-classification-benchmark)             | KNN, Decision Tree, Random Forest, AdaBoost, PCA, feature selection |
| 03  | [Clustering Comparison](./03-clustering-comparison)                   | K-means, DBSCAN                                                     |
| 04  | [Anomaly Detection](./04-anomaly-detection)                           | Isolation Forest, Local Outlier Factor                              |
| 05  | [Perceptron vs MLP](./05-perceptron-vs-mlp)                           | Perceptron, MLP, activation functions                               |
| 06  | [GA Knapsack](./06-ga-knapsack)                                       | Genetic Algorithm, selection, crossover, mutation                   |
| 07  | [Evolutionary Feature Selection](./07-evolutionary-feature-selection) | Filter GA, Wrapper GA, NSGA-II                                      |
| 08  | [GP Symbolic Regression](./08-gp-symbolic-regression)                 | Genetic Programming, expression trees, symbolic regression          |
| 09  | [Resource Allocation](./09-resource-allocation)                       | First Fit, Best Fit, VM/PM allocation                               |
| 10  | [Dispatching Rules](./10-dispatching-rules)                           | FCFS, SPT, EDD, job scheduling                                      |

Each project contains its own README with implementation details, experiment settings, and instructions.

## Topics Covered

### Machine Learning

- Classification
- Clustering
- Anomaly detection
- Feature selection
- Dimensionality reduction
- Neural networks
- Model evaluation

### Evolutionary Computation

- Genetic Algorithms
- Genetic Programming
- NSGA-II
- Multi-objective optimization
- Evolutionary feature selection

### Optimization and Scheduling

- Knapsack optimization
- Resource allocation
- Job scheduling
- First Fit and Best Fit heuristics
- Dispatching rules

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

Some algorithms are implemented from scratch to explore their internal mechanics, while others use established libraries for experimentation and comparison.

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd ai-ml-demo-lab
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Each demo can be run independently.

For example:

```bash
cd 01-decision-tree-from-scratch
python decision_tree.py data/rtg_A.csv assets/output_tree_A.txt
```

or:

```bash
cd 06-ga-knapsack
python knapsack.py
```

See the README inside each project directory for specific instructions.

## Repository Structure

```text
ai-ml-demo-lab/
├── 01-decision-tree-from-scratch/
├── 02-classification-benchmark/
├── 03-clustering-comparison/
├── 04-anomaly-detection/
├── 05-perceptron-vs-mlp/
├── 06-ga-knapsack/
├── 07-evolutionary-feature-selection/
├── 08-gp-symbolic-regression/
├── 09-resource-allocation/
├── 10-dispatching-rules/
├── .gitignore
├── README.md
└── requirements.txt
```

## Purpose

This repository serves as a practical AI/ML learning lab and portfolio of algorithm implementations.

The projects progress from fundamental machine learning techniques to evolutionary computation and optimization, with an emphasis on:

- implementing core algorithmic ideas
- comparing different approaches
- experimenting with hyperparameters
- evaluating model behaviour
- understanding trade-offs between different methods
