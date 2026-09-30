import numpy as np

from objectives import rosenbrock, griewank


def initialize_population(
    population_size,
    dimension,
    lower_bound,
    upper_bound,
    rng,
):
    population = rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(population_size, dimension),
    )

    return population


def evaluate_population(
    population,
    objective_function,
):
    costs = []

    for individual in population:
        cost = objective_function(individual)
        costs.append(cost)

    return np.asarray(costs, dtype=float)


def create_mutant_vector(
    population,
    target_index,
    scaling_factor,
    lower_bound,
    upper_bound,
    rng,
):
    population_size = len(population)

    all_indices = np.arange(population_size)
    candidate_indices = all_indices[all_indices != target_index]

    r1, r2, r3 = rng.choice(
        candidate_indices,
        size=3,
        replace=False,
    )

    x_r1 = population[r1]
    x_r2 = population[r2]
    x_r3 = population[r3]

    mutant = x_r1 + scaling_factor * (x_r2 - x_r3)

    mutant = np.clip(
        mutant,
        lower_bound,
        upper_bound,
    )

    return mutant


def crossover(
    target,
    mutant,
    crossover_rate,
    rng,
):
    dimension = len(target)

    trial = target.copy()
    # Randomly select a dimension j_rand and ensure that it always uses the value of the mutant.
    j_rand = rng.integers(dimension)

    for j in range(dimension):
        if rng.random() < crossover_rate or j == j_rand:
            trial[j] = mutant[j]

    return trial


# If the offspring is better than the parent, then replace "offspring" with another word; otherwise, keep "parent".
def select(
    target,
    target_cost,
    trial,
    trial_cost,
):
    if trial_cost <= target_cost:
        return trial.copy(), trial_cost

    return target.copy(), target_cost


def run_de(
    objective_function,
    population_size,
    dimension,
    lower_bound,
    upper_bound,
    scaling_factor,
    crossover_rate,
    generations,
    seed,
    verbose=True,
):
    if population_size < 4:
        raise ValueError("population_size must be at least 4.")

    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive.")

    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    population = initialize_population(
        population_size=population_size,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rng=rng,
    )

    costs = evaluate_population(
        population,
        objective_function,
    )

    best_index = np.argmin(costs)

    best_solution = population[best_index].copy()

    best_cost = float(costs[best_index])

    history = [best_cost]

    if verbose:
        print(
            "Initial best cost:",
            best_cost,
        )

    for generation in range(generations):

        next_population = population.copy()
        next_costs = costs.copy()

        for target_index in range(population_size):
            target = population[target_index]

            target_cost = costs[target_index]

            mutant = create_mutant_vector(
                population=population,
                target_index=target_index,
                scaling_factor=scaling_factor,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                rng=rng,
            )

            # Each of the mutated components in the mutant may not be adopted; the components that are not adopted will continue to use the original value of the target.
            trial = crossover(
                target=target,
                mutant=mutant,
                crossover_rate=crossover_rate,
                rng=rng,
            )

            trial_cost = objective_function(trial)

            selected, selected_cost = select(
                target=target,
                target_cost=target_cost,
                trial=trial,
                trial_cost=trial_cost,
            )

            next_population[target_index] = selected

            next_costs[target_index] = selected_cost

        population = next_population
        costs = next_costs

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
                "Best cost:",
                best_cost,
            )

    return (
        best_solution,
        best_cost,
        np.asarray(
            history,
            dtype=float,
        ),
    )


if __name__ == "__main__":

    best_solution, best_cost, history = run_de(
        objective_function=griewank,
        population_size=50,
        dimension=20,
        lower_bound=-30,
        upper_bound=30,
        scaling_factor=0.5,
        crossover_rate=0.9,
        generations=100,
        seed=42,
        verbose=True,
    )

    print("\nBest cost:")
    print(best_cost)

    print("\nBest solution:")
    print(best_solution)

    print("\nHistory length:")
    print(len(history))
