# Read data → Generate solution → Evolve the solution → Record results


import time
import random
import math
import copy
import statistics
import numpy as np


# 1. Target function
def target_function(x):
    if x > 0:
        return 1 / x + math.sin(x)
    else:
        return 2 * x + x**2 + 3.0


def generate_fitness_cases():
    xs = np.linspace(-5, 5, 201)
    ys = []

    for x in xs:
        y = target_function(float(x))
        ys.append(y)

    return xs, ys


# 2. Expression tree node
#             +
#  x+3 -->   / \  ---> Node("+", [Node("x"), Node(3.0)])
#           x   3
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def clone(self):
        return copy.deepcopy(self)

    # leaf eg: x,3.0,-2.5
    def is_terminal(self):
        return len(self.children) == 0

    def evaluate(self, x):
        try:
            if self.value == "x":
                return x

            if self.is_terminal():
                return float(self.value)

            if self.value == "+":
                result = self.children[0].evaluate(x) + self.children[1].evaluate(x)

            elif self.value == "-":
                result = self.children[0].evaluate(x) - self.children[1].evaluate(x)

            elif self.value == "*":
                result = self.children[0].evaluate(x) * self.children[1].evaluate(x)

            elif self.value == "/":
                left = self.children[0].evaluate(x)
                right = self.children[1].evaluate(x)

                # 1e‑6 = 0.000001
                if abs(right) < 1e-6:
                    result = 1.0
                else:
                    result = left / right

            # sin and cos only have one child
            elif self.value == "sin":
                result = math.sin(self.children[0].evaluate(x))

            elif self.value == "cos":
                result = math.cos(self.children[0].evaluate(x))

            else:
                result = 1e6

            # valid whether normal number
            if not math.isfinite(result):
                return 1e6
            # avoid too large or too small number
            if result > 1e6:
                return 1e6

            if result < -1e6:
                return -1e6

            return result

        except Exception:
            return 1e6

    # caculate how many operators x+3 = 3
    def size(self):
        total_size = 1

        for child in self.children:
            total_size += child.size()

        return total_size

    def depth(self):
        if self.is_terminal():
            return 1

        child_depths = []

        for child in self.children:
            child_depths.append(child.depth())

        return 1 + max(child_depths)

    def __str__(self):
        if self.is_terminal():
            if self.value == "x":
                return "x"
            return f"{self.value:.3f}"

        if self.value in ["+", "-", "*", "/"]:
            return f"({self.children[0]} {self.value} {self.children[1]})"

        if self.value in ["sin", "cos"]:
            return f"{self.value}({self.children[0]})"

        return str(self.value)


# which functions are allowed
FUNCTION_SET = ["+", "-", "*", "/", "sin", "cos"]


# 2.1 random create a terminal
def random_terminal():
    if random.random() < 0.5:
        return Node("x")
    else:
        return Node(random.uniform(-5, 5))


# 2.2 generate a whole tree
def random_tree(max_depth, method="grow"):
    # reach max depth
    if max_depth == 0:
        return random_terminal()

    if method == "grow" and random.random() < 0.3:
        return random_terminal()

    # random select a function
    function = random.choice(FUNCTION_SET)

    if function in ["+", "-", "*", "/"]:
        left_child = random_tree(max_depth - 1, method)
        right_child = random_tree(max_depth - 1, method)
        return Node(function, [left_child, right_child])

    if function in ["sin", "cos"]:
        child = random_tree(max_depth - 1, method)
        return Node(function, [child])

    return random_terminal()


# 3. Initialise population
def initialise_population(population_size, max_depth):
    population = []

    for i in range(population_size):
        depth = random.randint(2, max_depth)
        # Ramped‑half‑and‑half ： half full, half grow
        if i % 2 == 0:
            method = "grow"
        else:
            method = "full"

        individual = random_tree(depth, method)
        population.append(individual)

    return population


# 4 Use loss function to calculate fitness: compute RMSE as fitness metric
def calculate_rmse(individual, xs, ys):
    total_error = 0

    # SSE - Sum of Squared Errors
    for x, y_true in zip(xs, ys):
        y_pred = individual.evaluate(x)
        total_error += (y_pred - y_true) ** 2
    # MSE - Mean Squared Error
    mse = total_error / len(xs)
    # RMSE - Root Mean Squared Error
    return math.sqrt(mse)


# 4.1 fitness (add penalty version)
def gp_fitness(individual, xs, ys):
    rmse = calculate_rmse(individual, xs, ys)
    # to avoid overly large trees, penalise overly large trees to reduce bloat
    parsimony_penalty = 0.001 * individual.size()

    return rmse + parsimony_penalty


# 4.2 Evaluate all individuals in the population
def evaluate(population, xs, ys):
    evaluations = []
    for individual in population:
        error = gp_fitness(individual, xs, ys)
        # evaluations: （score, gp_tree）
        evaluations.append((error, individual))

    return evaluations


# 5. Parent selection: tournament selection
def tournament_selection(population, evaluations, k):
    # randomly select k "indices"
    candidate_indices = random.sample(range(len(population)), k)
    # ✅️In GP, lower RMSE/MSE means better performance.>
    winner_index = min(candidate_indices, key=lambda i: evaluations[i][0])

    # shallow copy
    return population[winner_index].clone()


def select_parent(population, evaluations, tournament_size):
    return tournament_selection(population, evaluations, tournament_size)


# 6. Crossover
def crossover(parent1, parent2, crossover_rate, max_depth):
    if random.random() > crossover_rate:
        return parent1.clone(), parent2.clone()

    child1 = subtree_crossover(parent1, parent2, max_depth)
    child2 = subtree_crossover(parent2, parent1, max_depth)

    return child1, child2


def subtree_crossover(parent1, parent2, max_depth):
    #     +        ←root path = ()
    #   /   \
    #  x     *     ←x: path=(0,)； *: path=(1,)
    #       / \
    #      y   5   ←y: path=(1,0)；5: path=(1,1)

    paths1 = all_paths(parent1)
    paths2 = all_paths(parent2)

    crossover_point1 = random.choice(paths1)
    crossover_point2 = random.choice(paths2)

    subtree_from_parent2 = get_subtree(parent2, crossover_point2)

    child = replace_subtree(parent1, crossover_point1, subtree_from_parent2)

    # If the child becomes too deep, reject it and return parent1.
    if child.depth() > max_depth:
        return parent1.clone()

    return child


# 6.1 Tree helper functions
def all_paths(node, current_path=()):
    """Return all possible paths in a tree.

    The root path is ().
    The first child is (0,).
    The second child is (1,).
    A deeper child can be (0, 1), etc.
    """
    paths = [current_path]

    for i, child in enumerate(node.children):
        child_paths = all_paths(child, current_path + (i,))
        paths.extend(child_paths)

    return paths


def get_subtree(node, path):
    """Get the subtree at a given path.
    ()       = root
    (0,)     = root.children[0]
    (1,)     = root.children[1]
    (1,0)    = root.children[1].children[0]
    (1,0,1)  = root.children[1].children[0].children[1]
    """
    current = node

    for index in path:
        current = current.children[index]

    return current


def replace_subtree(node, path, new_subtree):
    """Replace the subtree at path with new_subtree."""
    if path == ():
        return new_subtree.clone()

    copied_tree = node.clone()
    parent = get_subtree(copied_tree, path[:-1])
    parent.children[path[-1]] = new_subtree.clone()

    return copied_tree


# 7. Mutation
def mutate(individual, mutation_rate, max_depth):
    if random.random() < mutation_rate:
        return subtree_mutation(individual, max_depth)

    return individual.clone()


def subtree_mutation(individual, max_depth):
    paths = all_paths(individual)
    mutation_point = random.choice(paths)

    new_subtree_depth = random.randint(0, 3)
    # The new subtree is generated using the grow method with a small maximum depth, so that mutation can introduce new structures while limiting excessive tree growth.
    new_subtree = random_tree(new_subtree_depth, method="grow")

    child = replace_subtree(individual, mutation_point, new_subtree)

    if child.depth() > max_depth:
        return individual.clone()

    return child


# 8. GP loop
def gp(
    xs,
    ys,
    population_size=100,
    max_generations=100,
    max_depth=8,
    crossover_rate=0.9,
    mutation_rate=0.15,
    tournament_size=5,
    elite_count=1,
    seed=None,
):
    random.seed(seed)
    np.random.seed(seed)

    population = initialise_population(population_size, max_depth)
    evaluations = evaluate(population, xs, ys)

    history = []

    for generation in range(max_generations + 1):
        best_index = min(range(len(population)), key=lambda i: evaluations[i][0])
        best_error = evaluations[best_index][0]
        best_individual = evaluations[best_index][1]

        history.append(best_error)

        if generation == 0 or generation % 10 == 0:
            print(
                f"Generation {generation:3d} | "
                f"Fitness = {best_error:.6f} | "
                f"Size = {best_individual.size()} | "
                f"Depth = {best_individual.depth()}"
            )

        if generation == max_generations:
            break

        # elitism: smaller error is better
        elite_indices = sorted(range(len(population)), key=lambda i: evaluations[i][0])[
            :elite_count
        ]

        new_population = [population[i].clone() for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = select_parent(population, evaluations, tournament_size)
            parent2 = select_parent(population, evaluations, tournament_size)

            child1, child2 = crossover(parent1, parent2, crossover_rate, max_depth)

            child1 = mutate(child1, mutation_rate, max_depth)
            child2 = mutate(child2, mutation_rate, max_depth)

            if len(new_population) < population_size:
                new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        evaluations = evaluate(population, xs, ys)

    best_index = min(range(len(population)), key=lambda i: evaluations[i][0])
    best_fitness, best_individual = evaluations[best_index]

    return best_fitness, best_individual, history


GP_PARAMS = {
    "population_size": 100,
    "max_generations": 100,
    "max_depth": 8,
    "crossover_rate": 0.9,
    "mutation_rate": 0.15,
    "tournament_size": 5,
    "elite_count": 1,
}


def run_one_gp(xs, ys, seed):
    start_time = time.time()

    best_fitness, best_individual, history = gp(xs, ys, seed=seed, **GP_PARAMS)

    end_time = time.time()
    computational_time = end_time - start_time

    result = {
        "seed": seed,
        "fitness": best_fitness,
        "expression": str(best_individual),
        "size": best_individual.size(),
        "depth": best_individual.depth(),
        "history": history,
        "time": computational_time,
    }

    return result


def run_gp_experiment():
    print("\n==================== GP Symbolic Regression ====================")

    xs, ys = generate_fitness_cases()

    seeds = [1, 2, 3]
    results = []

    for run_number, seed in enumerate(seeds, start=1):
        print(f"\nRun {run_number}, seed={seed}")

        result = run_one_gp(xs, ys, seed)
        results.append(result)

        print(f"Final fitness: {result['fitness']:.6f}")
        print(f"Tree size: {result['size']}")
        print(f"Tree depth: {result['depth']}")
        print(f"Best expression: {result['expression']}")
        print(f"Time: {result['time']:.4f}s")

    print_summary_table(results)

    return results


def print_summary_table(results):
    fitness_values = [result["fitness"] for result in results]
    times = [result["time"] for result in results]
    sizes = [result["size"] for result in results]
    depths = [result["depth"] for result in results]

    print("\nSummary Table")
    print("Run | Seed | Fitness | Size | Depth | Time")

    for i, result in enumerate(results, start=1):
        print(
            f"{i} | "
            f"{result['seed']} | "
            f"{result['fitness']:.6f} | "
            f"{result['size']} | "
            f"{result['depth']} | "
            f"{result['time']:.4f}s"
        )

    print(
        f"Mean | - | "
        f"{statistics.mean(fitness_values):.6f} | "
        f"{statistics.mean(sizes):.2f} | "
        f"{statistics.mean(depths):.2f} | "
        f"{statistics.mean(times):.4f}s"
    )

    if len(results) > 1:
        print(
            f"Std | - | "
            f"{statistics.stdev(fitness_values):.6f} | "
            f"{statistics.stdev(sizes):.2f} | "
            f"{statistics.stdev(depths):.2f} | "
            f"{statistics.stdev(times):.4f}s"
        )


def main():
    run_gp_experiment()


if __name__ == "__main__":
    main()
