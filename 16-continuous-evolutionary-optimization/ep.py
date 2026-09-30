import numpy as np

from objectives import rosenbrock


def initialize_population(
    population_size, dimension, lower_bound, upper_bound, rng, initial_eta
):
    population = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(population_size, dimension),
    )

    mutation_levels = np.full(
        shape=(population_size, dimension),
        fill_value=initial_eta,
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


def mutate_mutation_levels(mutation_levels, tau, tau_prime, rng):
    population_size, dimension = mutation_levels.shape

    # Each individual receives one global noise value and one local value per gene.
    global_noise = rng.normal(loc=0, scale=1, size=(population_size, 1))
    local_noise = rng.normal(loc=0, scale=1, size=(population_size, dimension))
    combined_noise = tau_prime * global_noise + tau * local_noise

    return mutation_levels * np.exp(combined_noise)


def mutate_population(population, mutation_levels, lower_bound, upper_bound, rng):
    solution_noise = rng.normal(loc=0, scale=1, size=population.shape)
    offspring = population + mutation_levels * solution_noise

    return np.clip(offspring, lower_bound, upper_bound)


def tournament_selection(
    combined_population,
    combined_mutation_levels,
    combined_costs,
    population_size,
    q,
    rng,
):
    wins = np.zeros(len(combined_costs), dtype=int)

    for candidate_index in range(len(combined_costs)):
        all_indices = np.arange(len(combined_costs))
        possible_opponents = all_indices[all_indices != candidate_index]
        opponent_indices = rng.choice(possible_opponents, size=q, replace=False)

        candidate_cost = combined_costs[candidate_index]
        opponent_costs = combined_costs[opponent_indices]
        wins[candidate_index] = np.sum(candidate_cost <= opponent_costs)

    # Rank by wins first and use cost as the tie-breaker.
    ranked_indices = np.lexsort((combined_costs, -wins))
    selected_indices = ranked_indices[:population_size]

    return (
        combined_population[selected_indices],
        combined_mutation_levels[selected_indices],
        combined_costs[selected_indices],
    )


def run_ep(
    objective_function,
    population_size,
    dimension,
    lower_bound,
    upper_bound,
    initial_eta,
    generations,
    q,
    seed,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    population, mutation_levels = initialize_population(
        population_size=population_size,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rng=rng,
        initial_eta=initial_eta,
    )

    costs = evaluate_population(population, objective_function)
    tau, tau_prime = calculate_learning_rates(dimension)
    history = [float(np.min(costs))]

    if verbose:
        print("Initial best cost:", np.min(costs))
        print("Population shape:", population.shape)

    for generation in range(generations):
        new_mutation_levels = mutate_mutation_levels(
            mutation_levels=mutation_levels,
            tau=tau,
            tau_prime=tau_prime,
            rng=rng,
        )

        offspring = mutate_population(
            population=population,
            mutation_levels=new_mutation_levels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rng=rng,
        )

        offspring_costs = evaluate_population(offspring, objective_function)
        combined_population = np.vstack((population, offspring))
        combined_mutation_levels = np.vstack((mutation_levels, new_mutation_levels))
        combined_costs = np.concatenate((costs, offspring_costs))

        population, mutation_levels, costs = tournament_selection(
            combined_population=combined_population,
            combined_mutation_levels=combined_mutation_levels,
            combined_costs=combined_costs,
            population_size=population_size,
            q=q,
            rng=rng,
        )

        history.append(float(np.min(costs)))

        if verbose:
            print("Generation:", generation + 1, "Best cost:", np.min(costs))

    best_index = np.argmin(costs)
    best_solution = population[best_index].copy()
    best_cost = float(costs[best_index])

    return best_solution, best_cost, np.asarray(history, dtype=float)


if __name__ == "__main__":
    best_solution, best_cost, history = run_ep(
        objective_function=rosenbrock,
        population_size=50,
        dimension=20,
        lower_bound=-30,
        upper_bound=30,
        initial_eta=3.0,
        generations=10,
        q=10,
        seed=0,
        verbose=False,
    )

    print("Best cost:", best_cost)
    print("History length:", len(history))
