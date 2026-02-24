import math
import numpy as np
from .utils import *
from warnings import warn
import matplotlib.pyplot as plt


class SelfOrganizingMap:
  lr_decay_functions = {
    'inverse_decay_to_zero': inverse_decay_to_zero,
    'linear_decay_to_zero': linear_decay_to_zero,
    'asymptotic_decay': asymptotic_decay
  }

  sig_decay_functions = {
    'inverse_decay_to_one': inverse_decay_to_one,
    'linear_decay_to_one': linear_decay_to_one,
    'asymptotic_decay': asymptotic_decay
  }

  neig_functions = {
    'gaussian': gaussian,
    'mexican_hat': mexican_hat,
    'bubble': bubble,
    'triangle': triangle
  }

  def __init__(self, num_clusters: int,
                sigma=0.5, learning_rate=0.5,
                num_iterations=10,
                neighborhood_function='gaussian',
                lr_decay_function='asymptotic_decay',
                sigma_decay_function='asymptotic_decay'):

    self.sigma = sigma
    self.learning_rate = learning_rate
    self.num_clusters = num_clusters
    self.num_iterations = num_iterations

    self.points = None
    self.cluster_centers = None
    self.grid_shape = None
    self.positions_nd = None
    
    if isinstance(lr_decay_function, str):
      if lr_decay_function not in self.lr_decay_functions:
        msg = '%s not supported. Functions available: %s'
        raise ValueError(msg % (lr_decay_function, ', '.join(self.lr_decay_functions.keys())))
        
      self._learning_rate_decay_function = self.lr_decay_functions[lr_decay_function]

    elif callable(lr_decay_function):
      self._learning_rate_decay_function = lr_decay_function

    if sigma_decay_function not in self.sig_decay_functions:
        msg = '%s not supported. Functions available: %s'
        raise ValueError(msg % (sigma_decay_function,', '.join(self.sig_decay_functions.keys())))

    self._sigma_decay_function = self.sig_decay_functions[sigma_decay_function]

    if neighborhood_function not in self.neig_functions:
      msg = '%s not supported. Functions available: %s'
      raise ValueError(msg % (neighborhood_function, ', '.join(self.neig_functions.keys())))

    if neighborhood_function in ['triangle','bubble'] and (divmod(sigma, 1)[1] != 0 or sigma < 1):
        warn('sigma should be an integer >=1 when triangle or bubble are used as neighborhood function')

    self.neighborhood_function = self.neig_functions[neighborhood_function]

  def _auto_grid_shape(self, n):
      """Finds the most square-ish grid that can hold n items."""
      rows = int(math.floor(math.sqrt(n)))
      cols = int(math.ceil(n / rows))
      return (rows, cols)

  def _assign_nd_positions(self, n, grid_shape):
      """Assigns n-dimensional positions to each of the n centers."""
      positions = []
      for idx in range(n):
          row = idx // grid_shape[1]
          col = idx % grid_shape[1]
          positions.append(np.array([row, col]))
      return np.array(positions)

  def _initialize_centers(self):
    min_vals = np.min(self.points, axis=0)
    max_vals = np.max(self.points, axis=0)
    
    cluster_centers = np.zeros((self.num_clusters, self.points.shape[1]))
    for idx, position in enumerate(self.positions_nd):
      for dim in range(self.points.shape[1]):
        cluster_centers[idx, dim] = (
          min_vals[dim] + (position[dim % len(position)] / (self.grid_shape[dim % len(self.grid_shape)] - 1)) * (max_vals[dim] - min_vals[dim])
          if self.grid_shape[dim % len(self.grid_shape)] > 1 else min_vals[dim]
        )
    return cluster_centers    

  def train(self, points: np.ndarray = None,
            initial_cluster_centers: np.ndarray = None):
    if points is not None:
      self.set_points(points)

    if initial_cluster_centers is not None:
      self.set_cluster_centers(initial_cluster_centers)
    elif self.cluster_centers is None:
      self.set_num_clusters(self.num_clusters)
      self.set_cluster_centers(self._initialize_centers(), True)

    iteration = 0
    num_points = len(self.points)

    while iteration < self.num_iterations:
      # Shuffle points to avoid ordering bias
      indices = np.arange(num_points)
      np.random.shuffle(indices)

      total_movement = 0.0

      for t in indices:
        # Vectorized winning center location
        m = np.argmin(np.linalg.norm(self.cluster_centers - self.points[t], axis=1))

        # Vectorized update
        learning_rate = self._learning_rate_decay_function(self.learning_rate, iteration, self.num_iterations)
        sigma = self._sigma_decay_function(self.sigma, iteration, self.num_iterations)
        center_diffs = self.positions_nd - self.positions_nd[m]

        # Vectorized neighborhood parameters
        neighborhood_parameters = np.array([self.neighborhood_function(diff, sigma) for diff in center_diffs])

        diff = self.points[t] - self.cluster_centers
        movements = learning_rate * neighborhood_parameters[:, np.newaxis] * diff
        self.cluster_centers += movements
        total_movement += np.linalg.norm(movements, axis=1).sum()

      if total_movement < 1e-5:
        break

      iteration += 1

  def plot(self, filename: str = 'som_plot.png'):
    if self.points.shape[1] != 2:
      raise ValueError("Plotting is only supported for 2D data.")
      
    fig, ax = plt.subplots()

    ax.scatter(self.points[:, 0], self.points[:, 1], c='blue', label='Data Points')
    ax.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], c='red', label='Cluster Centers')
    ax.set_title(f'Self-Organizing Map')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.175), ncol=2)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

  def set_points(self, points: np.ndarray):
    self.points = points

  def set_cluster_centers(self, cluster_centers: np.ndarray, initialize: bool = False):
    if not initialize and self.num_clusters != len(cluster_centers):
      self.set_num_clusters(len(cluster_centers))
    self.cluster_centers = cluster_centers

  def set_num_clusters(self, num_clusters: int):
    self.num_clusters = num_clusters
    self.grid_shape = self._auto_grid_shape(self.num_clusters)
    self.positions_nd = self._assign_nd_positions(self.num_clusters, self.grid_shape)
    self.cluster_centers = None

  def get_cluster_centers(self):
    return self.cluster_centers
