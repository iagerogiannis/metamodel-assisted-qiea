import random
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from ..utils import custom_zip, flatten


def get_fitness(ind, accessor=None):
    if isinstance(ind, dict):
        return ind["fitness_score"]

    if accessor:
        return accessor(ind)

    return ind.fitness_score


def dominates(x, y):
    return all(a <= b for a, b in zip(x, y))


def non_dominated_sorting(population: list, accessor=None):
    fronts = []

    while len(population) > 0:
        front = []
        front_indices = []

        for i, d in enumerate(population):
            for j, e in enumerate(population):
                if i == j:
                    continue

                d_fitness = get_fitness(d, accessor)
                e_fitness = get_fitness(e, accessor)

                if all(a == b for a, b in zip(d_fitness, e_fitness)):
                    continue

                if dominates(e_fitness, d_fitness):
                    break

            else:
                front.append(d)
                front_indices.append(i)

        for i in reversed(front_indices):
            population.pop(i)

        fronts.append(front)

    return fronts


def crowding_distance_sorting(front: list, accessor=None):
    num_objectives = len(get_fitness(front[0], accessor))
    distances = [0] * len(front)

    # Individuals along with their distances
    augmented_pop = custom_zip(front, distances)

    for i in range(num_objectives):
        augmented_pop.sort(key=lambda x: get_fitness(x[0], accessor)[i])
        augmented_pop[0][1] = augmented_pop[-1][1] = float("inf")

        for j in range(1, len(front) - 1):
            augmented_pop[j][1] += (
                get_fitness(augmented_pop[j + 1][0], accessor)[i]
                - get_fitness(augmented_pop[j - 1][0], accessor)[i]
            )

    augmented_pop.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in augmented_pop]


def mo_sort(population: list, accessor=None):
    fronts = non_dominated_sorting(population[:], accessor)
    return flatten([crowding_distance_sorting(front, accessor) for front in fronts])


def get_pareto_front(population: list, accessor=None):
    return non_dominated_sorting(population[:], accessor)[0]


def compute_averaged_convergence(filepaths, n_points=500):
    dfs = [pd.read_csv(f"{fp}/out.csv") for fp in filepaths]

    min_eval = max(df["Num of Evals"].min() for df in dfs)
    max_eval = min(df["Num of Evals"].max() for df in dfs)
    common_evals = np.linspace(min_eval, max_eval, n_points)

    interpolated = []
    for df in dfs:
        f = interp1d(
            df["Num of Evals"],
            df["Fitness Score"],
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated.append(f(common_evals))

    return common_evals, np.mean(interpolated, axis=0)


def probabilistic_tournament_selection(arr, k, key=lambda x: x):
    """
    Performs probabilistic tournament selection on an array.

    Args:
        arr: List of elements to select from
        k: Tournament size (number of candidates per tournament)
        key: Optional function to extract comparison value (default: identity function)
             Use this to specify fitness criteria, e.g., lambda x: x.fitness_score

    Returns:
        Selected element from the tournament
    """
    import random

    # Handle edge cases
    if not arr:
        return None
    if k <= 0:
        return random.choice(arr)
    if k >= len(arr):
        return min(arr, key=key)

    # Select k random candidates from the array
    tournament = random.sample(arr, k)

    # Return the best (minimum) candidate from the tournament based on the key
    return min(tournament, key=key)


def linear_weighted_choice(arr, n=1, key=lambda x: x):
    """
    Randomly pick n unique elements from arr with linearly decreasing probability.
    First element (with lowest key value) has highest probability, last element (with highest key value) has lowest.

    Parameters:
    -----------
    arr : list
        The array to pick from
    n : int
        Number of unique choices to make (without replacement)
    key : function
        Optional function to extract comparison value (default: identity function)

    Returns:
    --------
    list of elements from arr
    """
    length = len(arr)
    n = min(n, length)  # Can't pick more unique items than exist

    if length == 0:
        return []

    if length == 1:
        return [arr[0]]

    arr = list(arr)  # Copy to avoid modifying original
    results = []

    for _ in range(n):
        # Sort arr by key so that weights are assigned from best to worst
        arr_sorted = sorted(arr, key=key)
        weights = [len(arr_sorted) - i for i in range(len(arr_sorted))]

        # Use random.choices to select one, but ensure uniqueness by removing after selection
        chosen = random.choices(arr_sorted, weights=weights, k=1)[0]
        results.append(chosen)
        arr.remove(chosen)  # Remove from arr so it can't be picked again

    return results


def linear_distribution(x, x1, y1, x2, y2, min_value=1, max_value=None):
    """
    Linearly interpolates a y-value at position x between two points (x1, y1) and (x2, y2),
    and clamps the result between min_value and max_value.

    Parameters:
    -----------
    x : float
        The x-value at which to interpolate.
    x1, y1 : float
        First point coordinates.
    x2, y2 : float
        Second point coordinates.
    min_value : int, optional
        Minimum value to return (default is 1).
    max_value : int or None, optional
        Maximum value to return (default is None, meaning no upper bound).

    Returns:
    --------
    int
        Interpolated y-value as an integer, clamped between min_value and max_value.
    """
    # Handle edge case where x1 == x2
    if x1 == x2:
        result = (y1 + y2) // 2

    else:
        result = y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    # Convert to integer and ensure it's at least the min_value
    result = max(min_value, int(round(result)))

    if max_value is not None:
        result = min(max_value, result)

    return result
