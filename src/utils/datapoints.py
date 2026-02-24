import numpy as np


def generate_datapoint(bounds, dim=2):
  """
  Generate a single random datapoint within the specified bounds.

  Parameters:
  - bounds: A tuple of (min, max) for the range of values.
  - dim: The number of dimensions for the datapoint (default is 2).

  Returns:
  - A numpy array representing the generated datapoint.
  """
  if len(bounds) != 2:
    raise ValueError("Bounds must be a tuple of (min, max)")

  if dim == 1:
    return np.random.uniform(bounds[0], bounds[1])

  return np.random.uniform(bounds[0], bounds[1], size=(dim,))


def generate_datapoints(num_of_points, bounds, dim=2):
  """
  Generate multiple random datapoints within the specified bounds.
  
  Parameters:
  - num_of_points: The number of datapoints to generate.
  - bounds: A tuple of (min, max) for the range of values.
  - dim: The number of dimensions for the datapoints (default is 2).
  
  Returns:
  - A numpy array of shape (num_of_points, dim) representing the generated datapoints.
  """
  if len(bounds) != 2:
    raise ValueError("Bounds must be a tuple of (min, max)")

  if dim == 1:
    return np.random.uniform(bounds[0], bounds[1], size=(num_of_points,))

  return np.random.uniform(bounds[0], bounds[1], size=(num_of_points, dim))
