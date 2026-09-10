# Genetic Programming for Symbolic Regression

A Genetic Programming (GP) implementation for symbolic regression, evolving mathematical expression trees to approximate a target function.

## Features

- Expression tree representation
- Ramped half-and-half initialization
- Tournament selection
- Subtree crossover
- Subtree mutation
- Elitism
- Maximum tree depth control
- Parsimony penalty to reduce tree bloat
- RMSE-based fitness evaluation
- Multiple-run evaluation

## Requirements

- Python 3.10+
- NumPy

```bash
pip install numpy
```

## Run

From the repository root:

```bash
cd 08-gp-symbolic-regression
python gp.py
```

## Experiments

The GP evolves expressions using the function set:

```text
+, -, *, /, sin, cos
```

The main parameters are:

```text
Population size: 100
Generations: 100
Maximum depth: 8
Crossover rate: 0.9
Mutation rate: 0.15
Tournament size: 5
Elites: 1
Runs: 3
```

Fitness is based on RMSE with a small parsimony penalty for larger expression trees.

## Results

The experiment is repeated using three random seeds.

For each run, the program records:

- Best fitness
- Best evolved expression
- Tree size and depth
- Computational time
- Fitness history

The generated results are stored in the `results/` directory.

## Project Structure

```text
08-gp-symbolic-regression/
├── results/
│   ├── best_program_seed_1.txt
│   ├── best_program_seed_2.txt
│   ├── best_program_seed_3.txt
│   ├── history_seed_1.csv
│   ├── history_seed_2.csv
│   ├── history_seed_3.csv
│   ├── report_table.csv
│   └── run_details.csv
├── gp.py
└── README.md
```

## Notes

The GP searches for mathematical expressions that approximate the target function over 201 fitness cases between `-5` and `5`.

Lower fitness values indicate better solutions.
