# Genetic Algorithm for Knapsack

A Genetic Algorithm (GA) for solving the 0/1 Knapsack Problem, with experiments comparing different selection and crossover strategies.

## Features

- Binary chromosome representation
- Fitness evaluation with capacity constraints
- Repair strategy for infeasible solutions
- Tournament and roulette-wheel selection
- One-point and uniform crossover
- Bit-flip mutation
- Elitism
- Multiple-run evaluation
- Convergence visualization

## Requirements

- Python 3.10+
- Matplotlib

```bash
pip install matplotlib
```

## Run

From the repository root:

```bash
cd 06-genetic-algorithm-knapsack
python knapsack.py
```

## Experiments

The GA is evaluated on three Knapsack instances:

```text
10_269
23_10000
100_995
```

Three GA configurations are compared:

```text
Original GA:
Tournament Selection + One-point Crossover

Modified GA 1:
Roulette Selection + One-point Crossover

Modified GA 2:
Tournament Selection + Uniform Crossover
```

The main GA parameters are:

```text
Population size: 100
Generations: 300
Crossover rate: 0.9
Mutation rate: 1 / number of items
Tournament size: 3
Elites: 2
Runs per experiment: 5
```

## Results

The experiments compare how selection and crossover strategies affect solution quality and convergence.

Known optimal values are:

| Dataset    | Optimal Value |
| ---------- | ------------: |
| `10_269`   |           295 |
| `23_10000` |          9767 |
| `100_995`  |          1514 |

Each configuration is evaluated across five random seeds, with the mean and standard deviation of the best solutions reported.

Convergence plots are also generated for each experiment.

## Project Structure

```text
06-genetic-algorithm-knapsack/
├── knapsack-data/
│   ├── 10_269
│   ├── 23_10000
│   └── 100_995
├── knapsack.py
└── README.md
```

## Notes

Overweight solutions are repaired by removing selected items with the lowest value-to-weight ratio first.

The GA uses elitism to preserve the best individuals between generations.
