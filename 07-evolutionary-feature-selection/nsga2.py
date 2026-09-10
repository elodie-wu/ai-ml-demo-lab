# Read data → Generate solution → Evolve the solution → Record results


import os
import time
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import statistics

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Read data ,skip_columns:skip clean1 data first two coloumn
def read_data(path, skip_columns=0):
    X = []
    y = []

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        data = line.replace(",", " ").split()
        data = data[skip_columns:]

        label = data[-1]
        features = [float(value) for value in data[:-1]]

        X.append(features)
        y.append(label)

    return X, y


# 2. Initialise population
def initialise_population(population_size, feature_count):
    """
    Randomly generate population_size individuals.
    Each individual is a 0/1 chromosome of length item_count.
    """
    population = []

    for _ in range(population_size):
        chromosome = []

        for _ in range(feature_count):
            gene = random.randint(0, 1)
            chromosome.append(gene)

        # at least select one feature
        if sum(chromosome) == 0:
            chromosome[random.randrange(feature_count)] = 1

        population.append(chromosome)

    return population


# 3. Data transformation
def transform_data(X, chromosome):
    select_indices = []
    # find all feature index
    for index, gene in enumerate(chromosome):
        if gene == 1:
            select_indices.append(index)

    new_X = []
    # Iterate and extract the columns corresponding to the indices of all features
    for row in X:
        new_row = []

        for index in select_indices:
            new_row.append(row[index])

        new_X.append(new_row)

    return new_X


# 4. Evaluate all individuals
def evaluate(population, X, y):
    evaluations = []

    for chromosome in population:
        error_rate, feature_ratio = calculate_objectives(chromosome, X, y)
        """crowding_distance： Crowding is used to rank individuals within the same non-dominated layer, preserving diversity and preventing the entire population from crowding onto a small Pareto front
        
        error_rate： If test set has a total of 100 samples, with 8 mispredicted. The error rate is 8 / 100 = 0.08.
        """
        evaluations.append(
            {
                "chromosome": chromosome[:],
                "error_rate": error_rate,
                "feature_ratio": feature_ratio,
                "rank": None,
                "crowding_distance": 0,
            }
        )
    return evaluations


# 5. Calculate two objectives
def calculate_objectives(chromosome, X, y):
    if sum(chromosome) == 0:
        error_rate = 1.0
    else:
        accuracy = classification_accuracy(X, y, chromosome)
        error_rate = 1 - accuracy

    feature_ratio = sum(chromosome) / len(chromosome)

    return error_rate, feature_ratio


# 6 classification_accuracy
def classification_accuracy(X, y, chromosome):
    X_selected = transform_data(X, chromosome)

    classifier = GaussianNB()
    classifier.fit(X_selected, y)

    predictions = classifier.predict(X_selected)
    accuracy = accuracy_score(y, predictions)

    return accuracy


# 7 dominates
def dominates(solution_a, solution_b):
    """If A's error is no worse than B's,
    A's feature ratio is no worse than B's,
    and A has at least one better objective,
    then A dominates B."""
    better_or_equal = (
        solution_a["error_rate"] <= solution_b["error_rate"]
        and solution_a["feature_ratio"] <= solution_b["feature_ratio"]
    )

    strictly_better = (
        solution_a["error_rate"] < solution_b["error_rate"]
        or solution_a["feature_ratio"] < solution_b["feature_ratio"]
    )

    return better_or_equal and strictly_better


# 8. non_dominated_sort: NSGA-II divides the solutions into different fronts. The 0th layer represents the best non-dominated solutions.
def non_dominated_sort(evaluations):
    """rank = 0: Not dominated by any solution, the best level.
    rank = 1: The next level after being influenced only by the rank 0 level.
    rank = 2: The next level after that."""
    for item in evaluations:
        item["rank"] = None
        item["crowding_distance"] = 0

    # initiate dominated_solutions： [[], [], [], ... , []]
    # p controls all individuals q (list of individuals)
    dominated_solutions = [[] for _ in range(len(evaluations))]
    # How many individuals control q
    domination_count = [0 for _ in range(len(evaluations))]

    # Layered Pareto Front ： fronts[0] = []
    fronts = [[]]

    # for(int i = 0; i < evaluations.length; i++){
    for p in range(len(evaluations)):
        for q in range(len(evaluations)):
            if p == q:
                continue

            # Each individual in the population must determine its dominance relationship with all other individuals.
            if dominates(evaluations[p], evaluations[q]):
                dominated_solutions[p].append(q)

            elif dominates(evaluations[q], evaluations[p]):
                domination_count[p] += 1
        # **No individual can control p** → p belongs to the first Pareto level (optimal frontier).
        if domination_count[p] == 0:
            evaluations[p]["rank"] = 0
            fronts[0].append(p)

    # 1. Processing begins from the first-layer fronts [0]
    current_front_index = 0
    # 2.As long as there are still individuals in the current layer, continue generating the next layer; once a layer becomes an empty list, the loop ends.
    while len(fronts[current_front_index]) > 0:
        next_front = []

        for p in fronts[current_front_index]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                # 3.Now that no individual can control q, q becomes a member of the new layer.
                if domination_count[q] == 0:
                    evaluations[q]["rank"] = current_front_index + 1
                    next_front.append(q)

        current_front_index += 1
        fronts.append(next_front)
    # 4.Remove empty lists at the end
    fronts.pop()

    return fronts


# 9. calculate_crowding_distance()
def calculate_crowding_distance(front, evaluations):
    if len(front) == 0:
        return

    for index in front:
        evaluations[index]["crowding_distance"] = 0
    # Two optimization objectives
    objectives = ["error_rate", "feature_ratio"]

    for objective in objectives:
        sorted_front = sorted(front, key=lambda i: evaluations[i][objective])

        # The boundary has no neighbors on both sides, so the formula cannot be calculated.
        evaluations[sorted_front[0]]["crowding_distance"] = float("inf")
        evaluations[sorted_front[-1]]["crowding_distance"] = float("inf")

        min_value = evaluations[sorted_front[0]][objective]
        max_value = evaluations[sorted_front[-1]][objective]

        # If all individuals have the same value for this objective, this dimension has no distinguishing power and can be skipped.
        if max_value == min_value:
            continue

        # Iterate through the middle individuals (skipping the first and last boundaries).
        for i in range(1, len(sorted_front) - 1):
            previous_value = evaluations[sorted_front[i - 1]][objective]
            next_value = evaluations[sorted_front[i + 1]][objective]

            # Normalized distance in the current target dimension
            distance = (next_value - previous_value) / (max_value - min_value)
            evaluations[sorted_front[i]]["crowding_distance"] += distance


def assign_rank_and_crowding(evaluations):
    fronts = non_dominated_sort(evaluations)

    for front in fronts:
        calculate_crowding_distance(front, evaluations)

    return fronts


# 10.tournament_selection:
def tournament_selection(population, evaluations, tournament_size):
    """First select the one with the smaller rank; if the ranks are the same, select the one with the larger crowding_distance."""
    candidate_indices = random.sample(range(len(population)), tournament_size)

    best_index = candidate_indices[0]

    for index in candidate_indices[1:]:
        candidate = evaluations[index]
        best = evaluations[best_index]

        # Rule 1: The player with the lower rank (Pareto level) wins immediately, and the new champion is updated.
        if candidate["rank"] < best["rank"]:
            best_index = index

        # Rule 2: If ranks are the same, the one with the greater crowding distance wins (better diversity).
        elif candidate["rank"] == best["rank"]:
            if candidate["crowding_distance"] > best["crowding_distance"]:
                best_index = index

    return population[best_index][:]


# 11. Crossover
def crossover(parent1, parent2, crossover_rate=0.9):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    cut_point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]

    return child1, child2


# 12. Mutation : randomly flip 0 and 1, mutation_rate = 1 / item_count
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    # at least select one feature
    if sum(chromosome) == 0:
        index = random.randrange(len(chromosome))
        chromosome[index] = 1
    return chromosome


# 13
def select_next_generation(combined_evaluations, population_size):
    fronts = assign_rank_and_crowding(combined_evaluations)

    new_evaluations = []

    for front in fronts:
        # if the whole front can be added
        if len(new_evaluations) + len(front) <= population_size:
            for index in front:
                new_evaluations.append(combined_evaluations[index])

        # if only part of this front can be added
        else:
            # ✅Within this layer: Sort by distance from crowding (prioritize sparse and diverse elements).
            sorted_front = sorted(
                front,
                key=lambda i: combined_evaluations[i]["crowding_distance"],
                reverse=True,
            )

            remaining_slots = population_size - len(new_evaluations)

            for index in sorted_front[:remaining_slots]:
                new_evaluations.append(combined_evaluations[index])
            # Once you've collected enough, exit the loop and discard any subsequent, even worse, leading edges.
            break

    new_population = []

    for item in new_evaluations:
        new_population.append(item["chromosome"][:])

    # recalculate rank and crowding distance for the new population
    assign_rank_and_crowding(new_evaluations)

    return new_population, new_evaluations


# 14 .nsga2
def nsga2(
    X,
    y,
    population_size=100,
    max_generations=30,
    crossover_rate=0.9,
    mutation_rate=None,
    tournament_size=2,
    seed=None,
):
    """In NSGA-II, each generation:

    The parent population generates the offspring population.

    Then the parent and offspring populations are merged.

    Then, the next generation is selected using rank and crowding distance."""
    random.seed(seed)

    feature_count = len(X[0])

    if mutation_rate is None:
        mutation_rate = 1 / feature_count

    population = initialise_population(population_size, feature_count)

    evaluations = evaluate(population, X, y)
    assign_rank_and_crowding(evaluations)

    for generation in range(1, max_generations + 1):
        offspring_population = []

        while len(offspring_population) < population_size:
            parent1 = tournament_selection(population, evaluations, tournament_size)
            parent2 = tournament_selection(population, evaluations, tournament_size)

            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            if len(offspring_population) < population_size:
                offspring_population.append(child1)

            if len(offspring_population) < population_size:
                offspring_population.append(child2)

        offspring_evaluations = evaluate(offspring_population, X, y)

        combined_evaluations = evaluations + offspring_evaluations

        population, evaluations = select_next_generation(
            combined_evaluations, population_size
        )

        if generation == 1 or generation % 10 == 0:
            first_front = [item for item in evaluations if item["rank"] == 0]

            best_error = min(item["error_rate"] for item in first_front)
            best_ratio = min(item["feature_ratio"] for item in first_front)

            print(
                f"Generation {generation} | "
                f"best_error={best_error:.4f}, "
                f"best_ratio={best_ratio:.4f}, "
                f"front_size={len(first_front)}"
            )

    final_fronts = assign_rank_and_crowding(evaluations)
    first_front = final_fronts[0]

    non_dominated_solutions = []

    for index in first_front:
        non_dominated_solutions.append(evaluations[index])

    return non_dominated_solutions


# 15 calculate_hypervolume:How much area do a set of non-dominated solutions cover on two targets
def calculate_hypervolume(solutions, reference_point=(1.0, 1.0)):
    points = []

    for solution in solutions:
        error = solution["error_rate"]
        ratio = solution["feature_ratio"]

        if error <= reference_point[0] and ratio <= reference_point[1]:
            points.append((error, ratio))

    points = sorted(set(points), key=lambda p: p[0])

    hypervolume = 0
    current_ratio = reference_point[1]

    for error, ratio in points:
        if ratio < current_ratio:
            width = reference_point[0] - error
            height = current_ratio - ratio
            hypervolume += width * height
            current_ratio = ratio

    return hypervolume


def plot_objective_space(dataset_name, run_number, solutions):
    errors = [solution["error_rate"] for solution in solutions]
    ratios = [solution["feature_ratio"] for solution in solutions]

    plt.figure()
    plt.scatter(errors, ratios)

    plt.xlabel("Classification error rate")
    plt.ylabel("Selected feature ratio")
    plt.title(f"{dataset_name} Run {run_number} Non-dominated Solutions")
    plt.grid(True)

    filename = f"{dataset_name}_run_{run_number}_objective_space.png"
    filename = filename.replace(" ", "_")

    plt.savefig(filename, dpi=300)
    plt.close()


def run_one_nsga2(X, y, dataset_name, run_number, seed):
    start_time = time.time()

    solutions = nsga2(X, y, seed=seed, **NSGA2_PARAMS)

    end_time = time.time()
    computational_time = end_time - start_time

    hypervolume = calculate_hypervolume(solutions, reference_point=(1.0, 1.0))

    plot_objective_space(dataset_name, run_number, solutions)

    result = {
        "seed": seed,
        "solutions": solutions,
        "hypervolume": hypervolume,
        "time": computational_time,
    }

    return result


def run_dataset_experiment(dataset_name, X, y):
    print(f"\n==================== {dataset_name} ====================")

    seeds = [1, 2, 3]
    results = []

    full_chromosome = [1 for _ in range(len(X[0]))]
    full_error = 1 - classification_accuracy(X, y, full_chromosome)

    print(f"Full feature set error rate: {full_error:.4f}")

    for run_number, seed in enumerate(seeds, start=1):
        print(f"\nRun {run_number}, seed={seed}")

        result = run_one_nsga2(X, y, dataset_name, run_number, seed)
        results.append(result)

        best_error = min(solution["error_rate"] for solution in result["solutions"])
        smallest_ratio = min(
            solution["feature_ratio"] for solution in result["solutions"]
        )

        print(f"Hypervolume: {result['hypervolume']:.6f}")
        print(f"Number of non-dominated solutions: {len(result['solutions'])}")
        print(f"Best error rate in solutions: {best_error:.4f}")
        print(f"Smallest feature ratio in solutions: {smallest_ratio:.4f}")
        print(f"Time: {result['time']:.4f}s")

    print_hypervolume_table(dataset_name, results)
    print_error_comparison_table(dataset_name, results, full_error)

    return results


def print_hypervolume_table(dataset_name, results):
    hypervolumes = [result["hypervolume"] for result in results]

    print(f"\n{dataset_name} Hypervolume Table")
    print("Run | Hypervolume")

    for i, result in enumerate(results, start=1):
        print(f"{i} | {result['hypervolume']:.6f}")

    print(f"Mean | {statistics.mean(hypervolumes):.6f}")
    print(f"Standard deviation | {statistics.stdev(hypervolumes):.6f}")


def print_error_comparison_table(dataset_name, results, full_error):
    print(f"\n{dataset_name} Error Comparison Table")
    print("Run | Full feature error | Best NSGA-II error | Smallest feature ratio")

    for i, result in enumerate(results, start=1):
        best_error = min(solution["error_rate"] for solution in result["solutions"])
        smallest_ratio = min(
            solution["feature_ratio"] for solution in result["solutions"]
        )

        print(
            f"{i} | "
            f"{full_error:.4f} | "
            f"{best_error:.4f} | "
            f"{smallest_ratio:.4f}"
        )


NSGA2_PARAMS = {
    "population_size": 100,
    "max_generations": 30,
    "crossover_rate": 0.9,
    "mutation_rate": None,
    "tournament_size": 2,
}


def main():
    VEHICLE_PATH = os.path.join(DATA_DIR, "vehicle", "vehicle.dat")
    CLEAN1_PATH = os.path.join(DATA_DIR, "musk", "clean1.data")

    X_vehicle, y_vehicle = read_data(VEHICLE_PATH, skip_columns=0)
    X_clean1, y_clean1 = read_data(CLEAN1_PATH, skip_columns=2)

    print("Vehicle:", len(X_vehicle), len(X_vehicle[0]), len(y_vehicle))
    print("Clean1:", len(X_clean1), len(X_clean1[0]), len(y_clean1))

    run_dataset_experiment("Vehicle", X_vehicle, y_vehicle)
    run_dataset_experiment("Clean1", X_clean1, y_clean1)


if __name__ == "__main__":
    main()
