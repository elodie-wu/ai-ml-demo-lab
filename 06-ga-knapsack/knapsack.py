# Read data → Generate solution → Evolve the solution → Record results


import os
import random
import statistics
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Read knapsack instance
def read_knapsack_data(path=None):
    """Read knapsack data from a text file.
    Returns:
        item_count
        capacity
        list of (value, weight) tuples.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    item_count, capacity = map(int, lines[0].split())

    items = []
    for line in lines[1:]:  # take items from second item
        value, weight = map(int, line.split())
        items.append((value, weight))

    return item_count, capacity, items


# 2. Initialise population
def initialise_population(population_size, item_count):
    """
    Randomly generate population_size individuals.
    Each individual is a 0/1 chromosome of length item_count.
    """
    population = []

    for _ in range(population_size):
        chromosome = []

        for _ in range(item_count):
            gene = random.randint(0, 1)
            chromosome.append(gene)

        population.append(chromosome)

    return population


# 3. Evaluate all individuals in the population
def evaluate(population, items, capacity):
    """
    Calculate the total value and total weight of each chromosome in the population.

    Returns:
        evaluations: [(total_value, total_weight, chromosome), ...]
    """
    evaluations = []

    for chromosome in population:
        total_value, total_weight = evaluate_one(chromosome, items)

        if total_weight > capacity:
            total_value, total_weight = repair(chromosome, items, capacity)

        evaluations.append((total_value, total_weight, chromosome))
    return evaluations


# 4.evaluate: caculate total value and weight from chromosome
def evaluate_one(chromosome, items):
    total_value = 0
    total_weight = 0
    for gene, (value, weight) in zip(chromosome, items):
        if gene == 1:
            total_value += value
            total_weight += weight
    return total_value, total_weight


# 5. repair : remove the lowest unit weight value
def repair(chromosome, items, capacity):
    """
    If the chromosome is overweight, remove the selected items
    until the total weight does not exceed the capacity.
    """
    total_value, total_weight = evaluate_one(chromosome, items)

    # find all indices from selected item
    selected_indices = []

    for i, gene in enumerate(chromosome):
        if gene == 1:
            selected_indices.append(i)

    # Sort by "value / weight" in ascending order.
    # Delete the item with the lowest "unit weight value" first.
    selected_indices.sort(key=lambda i: items[i][0] / items[i][1])

    for i in selected_indices:
        if total_weight <= capacity:
            break

        value, weight = items[i]

        chromosome[i] = 0  # cancel the selection of this item
        total_value -= value
        total_weight -= weight

    return total_value, total_weight


# 6. Parent selection: tournament selection
def tournament_selection(population, evaluations, k):
    # randomly select k "indices"
    candidate_indices = random.sample(range(len(population)), k)
    # select the index with the largest total_value among these k
    winner_index = max(candidate_indices, key=lambda i: evaluations[i][0])

    # shallow copy
    return population[winner_index][:]


# 6.1 Parent selection:developed GA-roulette_selection
def roulette_selection(population, evaluations):
    total_fitness = sum(evaluation[0] for evaluation in evaluations)

    if total_fitness == 0:
        return random.choice(population)[:]

    random_value = random.uniform(0, total_fitness)
    current_fitness = 0

    for chromosome, evaluation in zip(population, evaluations):
        current_fitness += evaluation[0]

        if current_fitness >= random_value:
            return chromosome[:]

    return population[-1][:]


# 6.2  Parent selection
def select_parent(population, evaluations, selection_method, tournament_size):
    if selection_method == "tournament":
        return tournament_selection(population, evaluations, tournament_size)

    if selection_method == "roulette":
        return roulette_selection(population, evaluations)

    raise ValueError("Unknown selection method")


# 7. Crossover
def crossover(parent1, parent2, crossover_rate=0.9, crossover_method="one_point"):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    if crossover_method == "one_point":
        cut_point = random.randint(1, len(parent1) - 1)

        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]

        return child1, child2

    if crossover_method == "uniform":
        child1 = []
        child2 = []

        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)

        return child1, child2

    raise ValueError("Unknown crossover method")


# 8. Mutation : randomly flip 0 and 1, mutation_rate = 1 / item_count
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]

    return chromosome


# 9.GA loop
def ga(
    path,
    population_size=100,
    max_generations=300,
    crossover_rate=0.9,
    mutation_rate=None,
    tournament_size=3,
    elite_count=2,
    seed=None,
    selection_method="tournament",
    crossover_method="one_point",
):
    random.seed(seed)
    item_count, capacity, items = read_knapsack_data(path)
    population = initialise_population(population_size, item_count)
    if mutation_rate is None:
        mutation_rate = 1 / item_count

    # record the best fiteness from each generation
    history = []
    evaluations = evaluate(population, items, capacity)

    for _ in range(max_generations):

        # reverse=True means sort in descending order
        elite_indices = sorted(
            range(len(population)), key=lambda i: evaluations[i][0], reverse=True
        )[:elite_count]

        new_population = [population[i][:] for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = select_parent(
                population, evaluations, selection_method, tournament_size
            )
            parent2 = select_parent(
                population, evaluations, selection_method, tournament_size
            )

            child1, child2 = crossover(
                parent1, parent2, crossover_rate, crossover_method
            )

            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            repair(child1, items, capacity)
            repair(child2, items, capacity)

            if len(new_population) < population_size:
                new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        evaluations = evaluate(population, items, capacity)
        best_value_this_generation = max(evaluation[0] for evaluation in evaluations)
        history.append(best_value_this_generation)

    # iterate through all evaluations and find the index with the best value.
    best_index = max(range(len(population)), key=lambda i: evaluations[i][0])
    best_value, best_weight, best_chromosome = evaluations[best_index]

    return best_value, best_weight, best_chromosome, history


def run_experiments(version_name, selection_method, crossover_method):
    instances = {
        "10_269": (os.path.join(DATA_DIR, "knapsack-data", "10_269"), 295),
        "23_10000": (os.path.join(DATA_DIR, "knapsack-data", "23_10000"), 9767),
        "100_995": (os.path.join(DATA_DIR, "knapsack-data", "100_995"), 1514),
    }

    seeds = [2, 4, 6, 8, 10]

    for instance_name, (data_path, optimal_value) in instances.items():
        print(f"\n======== {version_name}: {instance_name} ========")

        results = []
        histories = []

        for run_number, seed in enumerate(seeds, start=1):
            best_value, best_weight, best_chromosome, history = ga(
                data_path,
                population_size=100,
                max_generations=300,
                crossover_rate=0.9,
                mutation_rate=None,
                tournament_size=3,
                elite_count=2,
                selection_method=selection_method,
                crossover_method=crossover_method,
                seed=seed,
            )

            results.append((run_number, seed, best_value, best_weight))
            histories.append(history)

            print(
                f"Run {run_number}, seed={seed}: "
                f"value={best_value}, weight={best_weight}\n"
                f"Best chromosome: {best_chromosome}"
            )

        values = [result[2] for result in results]
        weights = [result[3] for result in results]

        print("\nRun | Seed | Total value | Total weight")

        for run_number, seed, value, weight in results:
            print(f"{run_number} | {seed} | {value} | {weight}")

        print(
            f"Mean | - | {statistics.mean(values):.2f} | "
            f"{statistics.mean(weights):.2f}"
        )
        print(
            f"Std | - | {statistics.stdev(values):.2f} | "
            f"{statistics.stdev(weights):.2f}"
        )

        print(f"Known optimal value: {optimal_value}")
        print(f"Best value found: {max(values)}")

        average_history = [
            statistics.mean(generation_values) for generation_values in zip(*histories)
        ]

        plt.figure()
        plt.plot(range(1, len(average_history) + 1), average_history)
        plt.xlabel("Generation")
        plt.ylabel("Average best fitness across five runs")
        plt.title(f"{version_name}: {instance_name}")
        plt.grid(True)
        plt.savefig(f"{version_name}_{instance_name}_convergence.png")
        plt.close()


def run_original_ga():
    run_experiments(
        version_name="Original GA",
        selection_method="tournament",
        crossover_method="one_point",
    )


def run_modified_gas():
    run_experiments(
        version_name="Modified GA 1 - Roulette Selection",
        selection_method="roulette",
        crossover_method="one_point",
    )

    run_experiments(
        version_name="Modified GA 2 - Uniform Crossover",
        selection_method="tournament",
        crossover_method="uniform",
    )


def main():
    run_original_ga()
    run_modified_gas()


if __name__ == "__main__":
    main()
