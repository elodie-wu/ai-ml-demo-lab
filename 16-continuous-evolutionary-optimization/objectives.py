import numpy as np


def rosenbrock(x):
    x_numpy = np.asarray(x, dtype=float)
    left = x_numpy[:-1]
    right = x_numpy[1:]

    terms = 100 * (left**2 - right) ** 2 + (left - 1) ** 2

    result = np.sum(terms)
    return float(result)


def griewank(x):
    x_numpy = np.asarray(x, dtype=float)

    sum_part = np.sum(x_numpy**2) / 4000

    indices = np.arange(1, len(x_numpy) + 1)
    print(f"indices: {indices}")

    scaled_x = x_numpy / np.sqrt(indices)
    cos_values = np.cos(scaled_x)
    print(f"cos_values: {cos_values}")
    product_part = np.prod(cos_values)

    result = sum_part - product_part + 1

    return float(result)
