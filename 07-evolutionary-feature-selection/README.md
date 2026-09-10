# Evolutionary Feature Selection

Evolutionary feature selection using Genetic Algorithms and NSGA-II to reduce feature sets while maintaining classification performance.

## Features

- Binary chromosome feature representation
- Filter-based Genetic Algorithm
- Wrapper-based Genetic Algorithm
- Mutual information feature scoring
- Gaussian Naive Bayes evaluation
- NSGA-II multi-objective optimization
- Pareto front and crowding distance
- Classification error vs feature ratio optimization
- Hypervolume evaluation
- Objective-space visualization

## Requirements

- Python 3.10+
- scikit-learn
- Matplotlib

```bash
pip install scikit-learn matplotlib
```

## Run

From the repository root:

```bash
cd 07-evolutionary-feature-selection
```

Run the GA feature selection experiments:

```bash
python feature_selection_ga.py
```

Run the NSGA-II experiments:

```bash
python nsga2.py
```

## Experiments

### Genetic Algorithm

Filter GA and Wrapper GA are compared on:

```text
WBCD
SONAR
```

The GA uses:

```text
Population size: 100
Generations: 100
Crossover rate: 0.9
Mutation rate: 1 / number of features
Tournament size: 3
Elites: 2
```

### NSGA-II

NSGA-II is evaluated on:

```text
Vehicle
Musk Clean1
```

Two objectives are minimized:

```text
Classification error
Selected feature ratio
```

The experiments compare the non-dominated solutions across three independent runs.

## Results

Filter GA and Wrapper GA demonstrate two different approaches to evolutionary feature selection.

NSGA-II produces a set of trade-off solutions between classification performance and the number of selected features.

Hypervolume and objective-space plots are used to evaluate the quality and diversity of the non-dominated solutions.

## Project Structure

```text
07-evolutionary-feature-selection/
├── assets/
│   ├── Clean1_run_1_objective_space.png
│   ├── Clean1_run_2_objective_space.png
│   ├── Clean1_run_3_objective_space.png
│   ├── Vehicle_run_1_objective_space.png
│   ├── Vehicle_run_2_objective_space.png
│   └── Vehicle_run_3_objective_space.png
├── musk/
│   └── clean1.data
├── sonar/
│   └── sonar.data
├── vehicle/
│   └── vehicle.dat
├── wbcd/
│   └── wbcd.data
├── feature_selection_ga.py
├── nsga2.py
└── README.md
```

## Notes

Each chromosome represents a feature subset, where `1` means the feature is selected and `0` means it is excluded.

NSGA-II minimizes both classification error and selected feature ratio to find multiple feature-selection trade-offs.
