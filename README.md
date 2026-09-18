# 🧠 AI/ML Demo Lab

A collection of small, self-contained implementations and experiments covering Artificial Intelligence, Machine Learning, Deep Learning, Evolutionary Computation, and Optimisation.

The repository focuses on understanding how algorithms work through practical implementations, comparisons, experiments, and visualisations rather than only using high-level APIs.

## 📚 Projects

| #   | Project                                                               | Topics                                                                 |
| --- | --------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 01  | [Decision Tree from Scratch](./01-decision-tree-from-scratch)         | Entropy, information gain, multiway splits                             |
| 02  | [Classification Benchmark](./02-classification-benchmark)             | KNN, Decision Tree, Random Forest, AdaBoost, PCA, feature selection    |
| 03  | [Clustering Comparison](./03-clustering-comparison)                   | K-means, DBSCAN                                                        |
| 04  | [Anomaly Detection](./04-anomaly-detection)                           | Isolation Forest, Local Outlier Factor                                 |
| 05  | [Perceptron vs MLP](./05-perceptron-vs-mlp)                           | Perceptron, MLP, activation functions                                  |
| 06  | [GA Knapsack](./06-ga-knapsack)                                       | Genetic Algorithm, selection, crossover, mutation, constraint handling |
| 07  | [Evolutionary Feature Selection](./07-evolutionary-feature-selection) | Filter GA, Wrapper GA, NSGA-II                                         |
| 08  | [GP Symbolic Regression](./08-gp-symbolic-regression)                 | Genetic Programming, expression trees, symbolic regression             |
| 09  | [Resource Allocation](./09-resource-allocation)                       | First Fit, Best Fit, VM/PM allocation                                  |
| 10  | [Dispatching Rules](./10-dispatching-rules)                           | FCFS, SPT, EDD, job scheduling                                         |

Projects 01–10 are present in the repository, each with its own README. Projects 11–19 are planned additions.

## 🔧 Planned Improvements to Existing Projects

These improvements extend the existing implementations and experiments.

| Project                             | Already implemented                                                                                                                                                  | Planned improvement                                                                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 06 · GA Knapsack                    | Tournament/roulette selection, one-point/uniform crossover, repair, elitism, three configurations, five seeds per instance, summary statistics and convergence plots | Optional notebook comparing mutation rates and elitism, with shared plots and optimality gaps                                                                     |
| 07 · Evolutionary Feature Selection | Filter GA, Wrapper GA, NSGA-II, Pareto ranking, crowding distance, hypervolume and objective-space plots                                                             | Prioritise cross-validation during feature selection and an independent test set; compare methods on shared datasets and evaluation settings                      |
| 08 · GP Symbolic Regression         | Full/Grow initialisation, subtree crossover and mutation, protected division, parsimony and repeated runs                                                            | Align saved results with the current script and standardise depth counting; optionally add prediction curves, tree visualisations and separate RMSE/fitness plots |

The current feature-selection scripts train and evaluate GaussianNB on the same data. Their reported accuracy and error describe training performance; independent evaluation is planned.

The GP directory includes saved results, but the current script does not export those files. Reproducible result export is part of the planned improvement.

Optional notebooks for existing projects will reuse the current algorithms and focus on experiments, visualisations and interpretation.

## 🗺️ Planned Roadmap

Projects 11–19 are not yet implemented. Future demos will primarily use one main Jupyter Notebook per topic.

| #   | Demo                                 | Topics                                                                  |
| --- | ------------------------------------ | ----------------------------------------------------------------------- |
| 11  | CNN Basics                           | Convolution, filters, activation functions, pooling, feature maps       |
| 12  | Autoencoder & VAE                    | Autoencoders, latent representations, reparameterisation, KL divergence |
| 13  | Generative Adversarial Networks      | GAN, DCGAN, conditional GANs, Pix2Pix, CycleGAN                         |
| 14  | Graph Search                         | BFS, DFS, graph traversal                                               |
| 15  | Adaptive Machine Learning            | Concept drift, stream learning, continual learning                      |
| 16  | Continuous Evolutionary Optimisation | EP, Fast EP, Evolution Strategies, Differential Evolution               |
| 17  | Ant Colony TSP                       | Ant Colony Optimisation, pheromone, evaporation, TSP                    |
| 18  | Particle Swarm Optimisation          | Particle movement, personal best, global best, swarm behaviour          |
| 19  | Estimation of Distribution           | UMDA, PBIL, cGA, probabilistic optimisation                             |

## 🧩 Topics Covered

| Area                         | Existing implementations                                                 | Planned additions                                                     |
| ---------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Machine Learning             | Classification, clustering, anomaly detection, PCA and feature selection | —                                                                     |
| Neural Networks              | Perceptron, MLP and activation comparisons                               | CNN, autoencoders, VAE and GANs                                       |
| Evolutionary Computation     | GA, GP and evolutionary feature selection                                | EP, Evolution Strategies, Differential Evolution and EDA              |
| Multi-Objective Optimisation | Pareto dominance, NSGA-II, crowding distance and hypervolume             | Improved feature-selection evaluation                                 |
| Optimisation and Scheduling  | Knapsack, resource allocation and dispatching rules                      | Graph search, Ant Colony Optimisation and Particle Swarm Optimisation |
| Adaptive Machine Learning    | —                                                                        | Concept drift, stream learning and continual learning                 |

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

Planned deep-learning demos may introduce PyTorch; it is not included in the current `requirements.txt`.

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

This abbreviated tree includes existing files and planned additions. Files marked `TODO` do not exist yet; `⭐` marks higher-priority planned work. The optional notebooks in 06 and 08 extend existing projects. Directories 11–19 are planned.

```text
ai-ml-demo-lab/
├── 01-decision-tree-from-scratch/
├── 02-classification-benchmark/
├── 03-clustering-comparison/
├── 04-anomaly-detection/
├── 05-perceptron-vs-mlp/
│
├── 06-ga-knapsack/
│   ├── knapsack.py
│   ├── knapsack-data/
│   ├── README.md
│   └── ga_knapsack_experiments.ipynb           # TODO optional: mutation and elitism experiments
│
├── 07-evolutionary-feature-selection/
│   ├── feature_selection_ga.py
│   ├── nsga2.py
│   ├── assets/
│   └── README.md
│
├── 08-gp-symbolic-regression/
│   ├── gp.py
│   ├── results/
│   ├── README.md
│   └── gp_symbolic_regression_demo.ipynb       # TODO optional: predictions and tree visualisations
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
