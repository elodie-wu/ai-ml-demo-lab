import numpy as np

from objectives import griewank


def initialize_positions(
    swarm_size,
    dimension,
    lower_bound,
    upper_bound,
    rng,
):
    return rng.uniform(
        low=lower_bound,
        high=upper_bound,
        size=(swarm_size, dimension),
    )


def initialize_velocities(
    swarm_size,
    dimension,
    max_velocity,
    rng,
):
    return rng.uniform(
        low=-max_velocity,
        high=max_velocity,
        size=(swarm_size, dimension),
    )


def evaluate_population(population, objective_function):
    costs = []

    for individual in population:
        cost = objective_function(individual)
        costs.append(cost)

    return np.asarray(costs, dtype=float)


def initialize_swarm(
    objective_function,
    swarm_size,
    dimension,
    lower_bound,
    upper_bound,
    max_velocity,
    rng,
):
    positions = initialize_positions(
        swarm_size=swarm_size,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rng=rng,
    )

    velocities = initialize_velocities(
        swarm_size=swarm_size,
        dimension=dimension,
        max_velocity=max_velocity,
        rng=rng,
    )

    costs = evaluate_population(positions, objective_function)
    personal_best_positions = positions.copy()
    personal_best_costs = costs.copy()
    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = float(personal_best_costs[global_best_index])

    return (
        positions,
        velocities,
        costs,
        personal_best_positions,
        personal_best_costs,
        global_best_position,
        global_best_cost,
    )


def update_velocities(
    positions,
    velocities,
    personal_best_positions,
    global_best_position,
    inertia_weight,
    cognitive_coefficient,
    social_coefficient,
    max_velocity,
    rng,
):
    r1 = rng.random(size=positions.shape)
    r2 = rng.random(size=positions.shape)

    inertia_component = inertia_weight * velocities
    cognitive_component = (
        cognitive_coefficient * r1 * (personal_best_positions - positions)
    )
    social_component = social_coefficient * r2 * (global_best_position - positions)
    new_velocities = inertia_component + cognitive_component + social_component

    return np.clip(new_velocities, -max_velocity, max_velocity)


def update_positions(
    positions,
    velocities,
    lower_bound,
    upper_bound,
):
    new_positions = positions + velocities

    return np.clip(new_positions, lower_bound, upper_bound)


def update_personal_bests(
    positions,
    costs,
    personal_best_positions,
    personal_best_costs,
):
    improved = costs < personal_best_costs
    personal_best_positions[improved] = positions[improved]
    personal_best_costs[improved] = costs[improved]

    return personal_best_positions, personal_best_costs


def update_global_best(
    personal_best_positions,
    personal_best_costs,
    global_best_position,
    global_best_cost,
):
    best_index = np.argmin(personal_best_costs)
    best_cost = float(personal_best_costs[best_index])

    if best_cost < global_best_cost:
        global_best_cost = best_cost
        global_best_position = personal_best_positions[best_index].copy()

    return global_best_position, global_best_cost


def run_pso(
    objective_function,
    swarm_size,
    dimension,
    lower_bound,
    upper_bound,
    max_velocity,
    inertia_weight,
    cognitive_coefficient,
    social_coefficient,
    generations,
    seed,
    verbose=True,
):
    if swarm_size <= 0:
        raise ValueError("swarm_size must be positive.")

    if max_velocity <= 0:
        raise ValueError("max_velocity must be positive.")

    if inertia_weight < 0:
        raise ValueError("inertia_weight must be non-negative.")

    if cognitive_coefficient < 0:
        raise ValueError("cognitive_coefficient must be non-negative.")

    if social_coefficient < 0:
        raise ValueError("social_coefficient must be non-negative.")

    rng = np.random.default_rng(seed)

    (
        positions,
        velocities,
        costs,
        personal_best_positions,
        personal_best_costs,
        global_best_position,
        global_best_cost,
    ) = initialize_swarm(
        objective_function=objective_function,
        swarm_size=swarm_size,
        dimension=dimension,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_velocity=max_velocity,
        rng=rng,
    )

    history = [global_best_cost]

    if verbose:
        print("Initial best cost:", global_best_cost)

    for generation in range(generations):
        velocities = update_velocities(
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            global_best_position=global_best_position,
            inertia_weight=inertia_weight,
            cognitive_coefficient=cognitive_coefficient,
            social_coefficient=social_coefficient,
            max_velocity=max_velocity,
            rng=rng,
        )

        positions = update_positions(
            positions=positions,
            velocities=velocities,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        costs = evaluate_population(positions, objective_function)
        personal_best_positions, personal_best_costs = update_personal_bests(
            positions=positions,
            costs=costs,
            personal_best_positions=personal_best_positions,
            personal_best_costs=personal_best_costs,
        )

        global_best_position, global_best_cost = update_global_best(
            personal_best_positions=personal_best_positions,
            personal_best_costs=personal_best_costs,
            global_best_position=global_best_position,
            global_best_cost=global_best_cost,
        )

        history.append(global_best_cost)

        if verbose:
            print("Generation:", generation + 1, "Best cost:", global_best_cost)

    return (
        global_best_position,
        global_best_cost,
        np.asarray(history, dtype=float),
    )


if __name__ == "__main__":
    best_solution, best_cost, history = run_pso(
        objective_function=griewank,
        swarm_size=50,
        dimension=20,
        lower_bound=-30,
        upper_bound=30,
        max_velocity=6,
        inertia_weight=0.7298,
        cognitive_coefficient=1.49618,
        social_coefficient=1.49618,
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

    print("\nHistory never gets worse:", np.all(np.diff(history) <= 0))
