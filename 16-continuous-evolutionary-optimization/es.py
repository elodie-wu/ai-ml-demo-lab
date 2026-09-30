import numpy as np

from objectives import rosenbrock


def initialize_population(
    population_size, dimension, lower_bound, upper_bound, rng, initial_sigma
):
    population = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(population_size, dimension),
    )

    mutation_levels = np.full(
        shape=(population_size, dimension),
        fill_value=initial_sigma,
        dtype=float,
    )

    return population, mutation_levels


def evaluate_population(population, objective_function):
    costs = []

    for individual in population:
        cost = objective_function(individual)
        costs.append(cost)

    return np.asarray(costs, dtype=float)


def calculate_learning_rates(dimension):
    tau = 1 / np.sqrt(2 * np.sqrt(dimension))
    tau_prime = 1 / np.sqrt(2 * dimension)

    return tau, tau_prime


def intermediate_global_recombination(population, mutation_levels, rng):
    parent_count, dimension = population.shape
    child = np.zeros(dimension, dtype=float)
    child_mutation_levels = np.zeros(dimension, dtype=float)

    for j in range(dimension):
        parent_indices = rng.choice(parent_count, size=2, replace=False)
        child[j] = np.mean(population[parent_indices, j])

        mutation_parent_indices = rng.choice(parent_count, size=2, replace=False)
        child_mutation_levels[j] = np.mean(
            mutation_levels[mutation_parent_indices, j]
        )

    return child, child_mutation_levels


def mutate_es_mutation_levels(child_mutation_levels, tau, tau_prime, rng):
    global_noise = rng.normal(loc=0, scale=1)
    local_noise = rng.normal(loc=0, scale=1, size=child_mutation_levels.shape)
    combined_noise = tau_prime * global_noise + tau * local_noise

    return child_mutation_levels * np.exp(combined_noise)


def mutate_es_solution(
    child,
    new_mutation_levels,
    lower_bound,
    upper_bound,
    rng,
):
    solution_noise = rng.normal(loc=0, scale=1, size=child.shape)
    mutated_child = child + new_mutation_levels * solution_noise

    return np.clip(mutated_child, lower_bound, upper_bound)


def generate_es_offspring(
    population,
    mutation_levels,
    offspring_count,
    tau,
    tau_prime,
    lower_bound,
    upper_bound,
    rng,
):
    dimension = population.shape[1]
    offspring_population = np.zeros((offspring_count, dimension), dtype=float)
    offspring_mutation_levels = np.zeros(
        (offspring_count, dimension),
        dtype=float,
    )

    for child_index in range(offspring_count):
        child, child_mutation_levels = intermediate_global_recombination(
            population,
            mutation_levels,
            rng,
        )

        new_child_mutation_levels = mutate_es_mutation_levels(
            child_mutation_levels,
            tau,
            tau_prime,
            rng,
        )

        offspring_population[child_index] = mutate_es_solution(
            child,
            new_child_mutation_levels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rng=rng,
        )
        offspring_mutation_levels[child_index] = new_child_mutation_levels

    return offspring_population, offspring_mutation_levels


def select_best_es_offspring(
    offspring_population,
    offspring_mutation_levels,
    offspring_costs,
    parent_count,
):
    ranked_indices = np.argsort(offspring_costs)
    selected_indices = ranked_indices[:parent_count]

    return (
        offspring_population[selected_indices].copy(),
        offspring_mutation_levels[selected_indices].copy(),
        offspring_costs[selected_indices].copy(),
    )


def run_es(
    objective_function,
    parent_count,
    offspring_count,
    dimension,
    lower_bound,
    upper_bound,
    initial_sigma,
    generations,
    seed,
    verbose=True,
):
    if parent_count < 2:
        raise ValueError("parent_count must be at least 2.")

    if offspring_count < parent_count:
        raise ValueError(
            "offspring_count must be greater than or equal to parent_count."
        )

    if initial_sigma <= 0:
        raise ValueError("initial_sigma must be positive.")

    rng = np.random.default_rng(seed)
    population, mutation_levels = initialize_population(
        population_size=parent_count,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rng=rng,
        initial_sigma=initial_sigma,
    )

    costs = evaluate_population(population, objective_function)
    tau, tau_prime = calculate_learning_rates(dimension)
    best_index = np.argmin(costs)
    best_solution = population[best_index].copy()
    best_cost = float(costs[best_index])
    history = [best_cost]

    if verbose:
        print("Initial best cost:", best_cost)
        print("Parent population shape:", population.shape)

    for generation in range(generations):
        offspring_population, offspring_mutation_levels = generate_es_offspring(
            population=population,
            mutation_levels=mutation_levels,
            offspring_count=offspring_count,
            tau=tau,
            tau_prime=tau_prime,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rng=rng,
        )

        offspring_costs = evaluate_population(
            offspring_population,
            objective_function,
        )

        population, mutation_levels, costs = select_best_es_offspring(
            offspring_population=offspring_population,
            offspring_mutation_levels=offspring_mutation_levels,
            offspring_costs=offspring_costs,
            parent_count=parent_count,
        )

        generation_best_index = np.argmin(costs)
        generation_best_cost = float(costs[generation_best_index])

        if generation_best_cost < best_cost:
            best_cost = generation_best_cost
            best_solution = population[generation_best_index].copy()

        history.append(best_cost)

        if verbose:
            print(
                "Generation:",
                generation + 1,
                "Best-so-far cost:",
                best_cost,
            )

    return best_solution, best_cost, np.asarray(history, dtype=float)


if __name__ == "__main__":
    best_solution, best_cost, history = run_es(
        objective_function=rosenbrock,
        parent_count=25,
        offspring_count=50,
        dimension=20,
        lower_bound=-30,
        upper_bound=30,
        initial_sigma=3.0,
        generations=10,
        seed=0,
        verbose=False,
    )

    print("Best cost:", best_cost)
    print("History length:", len(history))
