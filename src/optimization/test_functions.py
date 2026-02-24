from math import cos, sin, exp, sqrt, e, pi


def sphere_function(*x):
    return sum([xi**2 for xi in x])


def ackley_function(*x, a=20, b=0.2, c=2 * pi):
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([cos(c * xi) for xi in x])
    return -a * exp(-b * sqrt(sum1 / n)) - exp(sum2 / n) + a + e


def griewank_function(*x):
    sum1 = sum([xi**2 for xi in x])
    prod1 = 1
    for i, xi in enumerate(x):
        prod1 *= cos(xi / sqrt(i + 1))
    return sum1 / 4000 - prod1 + 1


def rastrigin_function(*x):
    return 10 * len(x) + sum([xi**2 - 10 * cos(2 * pi * xi) for xi in x])


def schwefel_function(*x):
    return 418.9829 * len(x) - sum([xi * sin(sqrt(abs(xi))) for xi in x])


def rosenbrock_function(*x):
    return sum(
        [100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)]
    )


def zdt1_function(*x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    h = 1 - sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]


def zdt2_function(*x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    h = 1 - (f1 / g) ** 2
    f2 = g * h
    return [f1, f2]


def zdt3_function(*x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    h = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1)
    f2 = g * h
    return [f1, f2]


def dtlz2_function(*x):
    n = len(x)
    g = sum([(xi - 0.5) ** 2 for xi in x[1:]])
    f1 = (1 + g) * cos(x[0] * pi / 2)
    f2 = (1 + g) * sin(x[0] * pi / 2)
    return [f1, f2]


def schaffer_n1_function(x):
    f1 = x**2
    f2 = (x - 2) ** 2
    return [f1, f2]


def get_test_function(test_function_name):
    match test_function_name:
        case "Sphere":
            return sphere_function
        case "Ackley":
            return ackley_function
        case "Griewank":
            return griewank_function
        case "Rastrigin":
            return rastrigin_function
        case "Schwefel":
            return schwefel_function
        case "Rosenbrock":
            return rosenbrock_function
        case "ZDT1":
            return zdt1_function
        case "ZDT2":
            return zdt2_function
        case "ZDT3":
            return zdt3_function
        case "DTLZ2":
            return dtlz2_function
        case "SchafferN1":
            return schaffer_n1_function
        case _:
            raise ValueError("Invalid test function name")


def get_true_pareto_front(func_name, n_points=100):
    """Generate true Pareto front for known test functions.

    Args:
        func_name: Name of the test function
        n_points: Number of points to generate on the front

    Returns:
        np.ndarray: Array of shape (n_points, n_objectives) or None if unknown
    """
    import numpy as np

    if func_name == "ZDT1":
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    elif func_name == "ZDT2":
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1**2
        return np.column_stack([f1, f2])

    elif func_name == "ZDT3":
        # ZDT3 has disconnected Pareto front
        f1_segments = []
        f2_segments = []

        # Define the connected regions (approximately)
        regions = [
            (0.0, 0.0830),
            (0.1822, 0.2577),
            (0.4093, 0.4538),
            (0.6183, 0.6525),
            (0.8233, 0.8518),
        ]

        points_per_region = n_points // len(regions)
        for start, end in regions:
            f1 = np.linspace(start, end, points_per_region)
            f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * pi * f1)
            f1_segments.extend(f1)
            f2_segments.extend(f2)

        return np.column_stack([f1_segments, f2_segments])

    elif func_name == "DTLZ2":
        # For 2 objectives: f1^2 + f2^2 = 1 (unit circle in objective space)
        theta = np.linspace(0, pi / 2, n_points)
        f1 = np.cos(theta)
        f2 = np.sin(theta)
        return np.column_stack([f1, f2])

    elif func_name == "SchafferN1":
        x = np.linspace(0, 2, n_points)
        f1 = x**2
        f2 = (x - 2) ** 2
        return np.column_stack([f1, f2])

    return None
