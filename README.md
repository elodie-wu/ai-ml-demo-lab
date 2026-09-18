# 🧠 AI/ML Demo Lab

A collection of small, self-contained implementations and experiments covering Artificial Intelligence, Machine Learning, Deep Learning, Evolutionary Computation, and Optimisation.

The repository focuses on understanding how algorithms work through practical implementations, comparisons, experiments, and visualisations rather than only using high-level APIs.

## 📚 Projects

| # | Project | Topics |
| --- | --- | --- |
| 01 | [Decision Tree from Scratch](./01-decision-tree-from-scratch) | Entropy, information gain, multiway splits |
| 02 | [Classification Benchmark](./02-classification-benchmark) | KNN, Decision Tree, Random Forest, AdaBoost, PCA, feature selection |
| 03 | [Clustering Comparison](./03-clustering-comparison) | K-means, DBSCAN |
| 04 | [Anomaly Detection](./04-anomaly-detection) | Isolation Forest, Local Outlier Factor |
| 05 | [Perceptron vs MLP](./05-perceptron-vs-mlp) | Perceptron, MLP, activation functions |
| 06 | [GA Knapsack](./06-ga-knapsack) | Genetic Algorithm, selection, crossover, mutation, constraint handling |
| 07 | [Evolutionary Feature Selection](./07-evolutionary-feature-selection) | Filter GA, Wrapper GA, NSGA-II |
| 08 | [GP Symbolic Regression](./08-gp-symbolic-regression) | Genetic Programming, expression trees, symbolic regression |
| 09 | [Resource Allocation](./09-resource-allocation) | First Fit, Best Fit, VM/PM allocation |
| 10 | [Dispatching Rules](./10-dispatching-rules) | FCFS, SPT, EDD, job scheduling |

Each completed project contains its own README with implementation details, experiment settings, and instructions.

## 🗺️ Planned Roadmap

Future demos will primarily use one main Jupyter Notebook per topic.

| # | Demo | Topics |
| --- | --- | --- |
| 11 | CNN Basics | Convolution, filters, activation functions, pooling, feature maps |
| 12 | Autoencoder & VAE | Autoencoders, latent representations, reparameterisation, KL divergence |
| 13 | Generative Adversarial Networks | GAN, DCGAN, conditional GANs, Pix2Pix, CycleGAN |
| 14 | Graph Search | BFS, DFS, graph traversal |
| 15 | Adaptive Machine Learning | Concept drift, stream learning, continual learning |
| 16 | Continuous Evolutionary Optimisation | EP, Fast EP, Evolution Strategies, Differential Evolution |
| 17 | Ant Colony TSP | Ant Colony Optimisation, pheromone, evaporation, TSP |
| 18 | Particle Swarm Optimisation | Particle movement, personal best, global best, swarm behaviour |
| 19 | Estimation of Distribution | UMDA, PBIL, cGA, probabilistic optimisation |

## 🧩 Topics Covered

### 📊 Machine Learning

- Classification
- Clustering
- Anomaly detection
- Feature selection
- Dimensionality reduction
- Neural networks
- Model evaluation

### 🖼️ Deep Learning

- Perceptrons and multilayer perceptrons
- Convolutional Neural Networks
- Autoencoders
- Variational Autoencoders
- Generative Adversarial Networks
- Image generation and translation

### 🧬 Evolutionary Computation

- Genetic Algorithms
- Genetic Programming
- Evolutionary Programming
- Evolution Strategies
- Differential Evolution
- Evolutionary feature selection
- Estimation of Distribution Algorithms

### 🎯 Multi-Objective Optimisation

- Pareto dominance
- Pareto fronts
- NSGA-II
- Feature selection with conflicting objectives

### 🐜 Swarm Intelligence

- Ant Colony Optimisation
- Particle Swarm Optimisation

### 🔍 Search, Optimisation and Scheduling

- Graph search
- Knapsack optimisation
- Resource allocation
- Job scheduling
- Dispatching rules

### 🔄 Adaptive Machine Learning

- Concept drift
- Stream learning
- Continual learning

## 📓 Notebook Style

Planned notebook-based demos are designed to remain self-contained.

A typical notebook follows this structure:

```text
1. Problem
2. Algorithm / Model
3. Implementation
4. Experiment
5. Visualisation
6. Results
7. Key Takeaways
```

The goal is to make each demo easy to follow while keeping the implementation focused on the core algorithmic ideas.

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook
- PyTorch

Additional libraries may be introduced for individual experiments.

Some algorithms are implemented from scratch to explore their internal mechanics, while others use established libraries for experimentation and comparison.

## 🚀 Setup

Clone the repository:

```bash
git clone <repository-url>
cd ai-ml-demo-lab
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For notebook-based demos:

```bash
jupyter notebook
```

or open the repository directly in VS Code with the Jupyter extension.

## ▶️ Run

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

Notebook-based demos can be opened and executed from top to bottom.

See the README inside each completed project directory for specific instructions.

## 🗂️ Repository Structure

`⭐` marks higher-priority planned work.

```text
ai-ml-demo-lab/
├── 01-decision-tree-from-scratch/
├── 02-classification-benchmark/
├── 03-clustering-comparison/
├── 04-anomaly-detection/
├── 05-perceptron-vs-mlp/
│
├── 06-ga-knapsack/
│   └── ga_knapsack_experiments.ipynb           # TODO ⭐
│
├── 07-evolutionary-feature-selection/
│   └── multi_objective_evolution.ipynb         # TODO ⭐
│
├── 08-gp-symbolic-regression/
│   └── gp_symbolic_regression_demo.ipynb       # TODO ⭐
│
├── 09-resource-allocation/
├── 10-dispatching-rules/
│
├── 11-cnn-basics/
│   └── cnn_basics.ipynb                        # TODO ⭐
│
├── 12-autoencoder-vae/
│   └── autoencoder_vae.ipynb                   # TODO ⭐
│
├── 13-generative-adversarial-networks/
│   └── generative_adversarial_networks.ipynb   # TODO ⭐
│
├── 14-graph-search-bfs-dfs/
│   └── graph_search_bfs_dfs.ipynb              # TODO ⭐
│
├── 15-adaptive-machine-learning/
│   └── adaptive_machine_learning.ipynb         # TODO ⭐
│
├── 16-continuous-evolutionary-optimization/
│   └── continuous_evolutionary_optimization.ipynb  # TODO ⭐
│
├── 17-ant-colony-tsp/
│   └── ant_colony_tsp.ipynb                    # TODO
│
├── 18-particle-swarm-optimization/
│   └── particle_swarm_optimization.ipynb       # TODO
│
├── 19-estimation-of-distribution/
│   └── estimation_of_distribution.ipynb        # TODO ⭐
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🎯 Purpose

This repository serves as a practical AI/ML learning lab and portfolio of algorithm implementations.

The projects progress from fundamental machine learning techniques to deep learning, evolutionary computation, optimisation, and adaptive learning, with an emphasis on:

- implementing core algorithmic ideas
- understanding how algorithms work internally
- comparing different approaches
- experimenting with hyperparameters
- visualising model and optimisation behaviour
- evaluating performance
- understanding trade-offs between different methods

The repository is intended to grow progressively as new techniques are explored and implemented.
