# Read data → Generate solution → Evolve the solution → Record results


import os
import time
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import statistics

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Read data
def read_data(path=None):
    """Read data from a text file.
    feature1, feature2, ..., featureN, label
    Returns:
        X,y
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    X = []
    y = []

    for line in lines:
        data = line.split(",")
        label = int(data[-1])
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


# 3. Data transformation that takes the original dataset and the selected features, and creates a new dataset with only the selected features (the corresponding subset of columns).
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


# 4. caculate accuracy
def classification_accuracy(X, y, chromosome):
    X_selected = transform_data(X, chromosome)

    classifier = GaussianNB()
    classifier.fit(X_selected, y)

    predictions = classifier.predict(X_selected)
    accuracy = accuracy_score(y, predictions)

    return accuracy


# 5.1 wrapper_fitness
def wrapper_fitness(chromosome, X, y):
    if sum(chromosome) == 0:
        return 0

    accuracy = classification_accuracy(X, y, chromosome)

    return accuracy


# 5.2 filter_fitness
def filter_fitness(chromosome, filter_scores):
    selected_scores = []

    for index, gene in enumerate(chromosome):
        if gene == 1:
            selected_scores.append(filter_scores[index])

    if len(selected_scores) == 0:
        return 0

    fitness = statistics.mean(selected_scores)

    return fitness


# 6. Evaluate all individuals in the population
def evaluate(population, X, y, ga_type, filter_scores=None):
    evaluations = []

    for chromosome in population:
        if ga_type == "filter":
            fitness = filter_fitness(chromosome, filter_scores)

        elif ga_type == "wrapper":
            fitness = wrapper_fitness(chromosome, X, y)

        else:
            raise ValueError("Unknown GA type")

        evaluations.append((fitness, chromosome))

    return evaluations


# 7.1 Parent selection: tournament selection
def tournament_selection(population, evaluations, k):
    # randomly select k "indices"
    candidate_indices = random.sample(range(len(population)), k)
    # select the index with the largest total_value among these k
    winner_index = max(candidate_indices, key=lambda i: evaluations[i][0])

    # shallow copy
    return population[winner_index][:]


# 7.2 Parent selection:developed GA-roulette_selection
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


# 7 Parent selection
def select_parent(population, evaluations, selection_method, tournament_size):
    if selection_method == "tournament":
        return tournament_selection(population, evaluations, tournament_size)

    if selection_method == "roulette":
        return roulette_selection(population, evaluations)

    raise ValueError("Unknown selection method")


# 8. Crossover
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


# 9. Mutation : randomly flip 0 and 1, mutation_rate = 1 / item_count
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    # at least select one feature
    if sum(chromosome) == 0:
        index = random.randrange(len(chromosome))
        chromosome[index] = 1
    return chromosome


# 10.GA loop
def ga(
    X,
    y,
    ga_type,
    population_size=100,
    max_generations=100,
    crossover_rate=0.9,
    mutation_rate=None,
    tournament_size=3,
    elite_count=2,
    seed=None,
    selection_method="tournament",
    crossover_method="one_point",
):
    random.seed(seed)

    feature_count = len(X[0])

    if mutation_rate is None:
        mutation_rate = 1 / feature_count

    # filter
    filter_scores = None

    if ga_type == "filter":
        filter_scores = mutual_info_classif(X, y, random_state=1)

    population = initialise_population(population_size, feature_count)

    evaluations = evaluate(population, X, y, ga_type, filter_scores)

    # record the best fiteness from each generation
    history = []

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

            if len(new_population) < population_size:
                new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        evaluations = evaluate(population, X, y, ga_type, filter_scores)
        best_value_this_generation = max(evaluation[0] for evaluation in evaluations)
        history.append(best_value_this_generation)

    # iterate through all evaluations and find the index with the best value.
    best_index = max(range(len(population)), key=lambda i: evaluations[i][0])
    best_fitness, best_chromosome = evaluations[best_index]

    selected_feature_indices = [
        index for index, gene in enumerate(best_chromosome) if gene == 1
    ]
    accuracy = classification_accuracy(X, y, best_chromosome)

    return best_fitness, best_chromosome, accuracy, selected_feature_indices, history


GA_PARAMS = {
    "population_size": 100,
    "max_generations": 100,
    "crossover_rate": 0.9,
    "mutation_rate": None,
    "tournament_size": 3,
    "elite_count": 2,
    "selection_method": "tournament",
    "crossover_method": "one_point",
}


def run_one_ga(X, y, ga_type, seed):
    start_time = time.time()

    best_fitness, best_chromosome, accuracy, selected_feature_indices, history = ga(
        X, y, ga_type, seed=seed, **GA_PARAMS
    )

    end_time = time.time()
    computational_time = end_time - start_time

    result = {
        "seed": seed,
        "fitness": best_fitness,
        "chromosome": best_chromosome,
        "accuracy": accuracy,
        "selected_features": selected_feature_indices,
        "history": history,
        "time": computational_time,
    }

    return result


def run_dataset_experiment(dataset_name, X, y):
    print(f"\n==================== {dataset_name} ====================")

    seeds = [2, 4, 6, 8, 10]

    filter_results = []
    wrapper_results = []

    for run_number, seed in enumerate(seeds, start=1):
        filter_result = run_one_ga(X, y, "filter", seed)
        wrapper_result = run_one_ga(X, y, "wrapper", seed)

        filter_results.append(filter_result)
        wrapper_results.append(wrapper_result)

        print(f"\nRun {run_number}, seed={seed}")

        print(
            f"FilterGA: "
            f"time={filter_result['time']:.4f}s, "
            f"fitness={filter_result['fitness']:.4f}, "
            f"accuracy={filter_result['accuracy']:.4f}, "
            f"selected_features={len(filter_result['selected_features'])}"
        )

        print(
            f"WrapperGA: "
            f"time={wrapper_result['time']:.4f}s, "
            f"fitness={wrapper_result['fitness']:.4f}, "
            f"accuracy={wrapper_result['accuracy']:.4f}, "
            f"selected_features={len(wrapper_result['selected_features'])}"
        )

    print_time_table(dataset_name, filter_results, wrapper_results)
    print_accuracy_table(dataset_name, filter_results, wrapper_results)

    return filter_results, wrapper_results


def print_accuracy_table(dataset_name, filter_results, wrapper_results):
    filter_accuracies = [result["accuracy"] for result in filter_results]
    wrapper_accuracies = [result["accuracy"] for result in wrapper_results]

    print(f"\n{dataset_name} Classification Accuracy Table")
    print("Run | FilterGA | WrapperGA")

    for i in range(len(filter_results)):
        print(
            f"{i + 1} | "
            f"{filter_accuracies[i]:.4f} | "
            f"{wrapper_accuracies[i]:.4f}"
        )

    print(
        f"Mean | "
        f"{statistics.mean(filter_accuracies):.4f} | "
        f"{statistics.mean(wrapper_accuracies):.4f}"
    )

    print(
        f"Standard deviation | "
        f"{statistics.stdev(filter_accuracies):.4f} | "
        f"{statistics.stdev(wrapper_accuracies):.4f}"
    )


def print_time_table(dataset_name, filter_results, wrapper_results):
    filter_times = [result["time"] for result in filter_results]
    wrapper_times = [result["time"] for result in wrapper_results]

    print(f"\n{dataset_name} Computational Time Table")
    print("Run | FilterGA | WrapperGA")

    for i in range(len(filter_results)):
        print(f"{i + 1} | " f"{filter_times[i]:.4f} | " f"{wrapper_times[i]:.4f}")

    print(
        f"Mean | "
        f"{statistics.mean(filter_times):.4f} | "
        f"{statistics.mean(wrapper_times):.4f}"
    )

    print(
        f"Standard deviation | "
        f"{statistics.stdev(filter_times):.4f} | "
        f"{statistics.stdev(wrapper_times):.4f}"
    )


def main():
    WBCD_PATH = os.path.join(DATA_DIR, "wbcd", "wbcd.data")
    SONAR_PATH = os.path.join(DATA_DIR, "sonar", "sonar.data")

    X_wbcd, y_wbcd = read_data(WBCD_PATH)
    X_sonar, y_sonar = read_data(SONAR_PATH)

    print("WBCD:", len(X_wbcd), len(X_wbcd[0]), len(y_wbcd))
    print("SONAR:", len(X_sonar), len(X_sonar[0]), len(y_sonar))

    run_dataset_experiment("WBCD", X_wbcd, y_wbcd)
    run_dataset_experiment("SONAR", X_sonar, y_sonar)


if __name__ == "__main__":
    main()
