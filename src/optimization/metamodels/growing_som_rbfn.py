import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .rbfn import RadialBasisFunctionNetwork
from .som import SelfOrganizingMap
from .mst import MinimumSpanningTree
from .radii_calculator import RadiiCalculator
import pandas as pd
import time


class GrowingSOMRBFNetwork:
  def __init__(self, minmax_num_clusters, test_size=0.25, max_retries=10, metamodel_type='simple_rbf'):
    self.max_retries = max_retries
    self.metamodel_type = metamodel_type

    self.mst = MinimumSpanningTree()
    self.som = SelfOrganizingMap(minmax_num_clusters[0]) if metamodel_type == 'growing_rbf' else None
    self.radii_calculator = RadiiCalculator()
    self.rbf = RadialBasisFunctionNetwork(interpolate=(metamodel_type == 'simple_rbf'))

    self.minmax_num_clusters = minmax_num_clusters if metamodel_type == 'growing_rbf' else None
    self.test_size = test_size

    self.cluster_centers = None
    self.radii = None
    self.training_patterns = None
    self.X_train, self.X_test, \
    self.y_train, self.y_test = None, None, None, None
    
    self.cluster_centers_opt = None
    self.radii_opt = None
    self.weights_opt = None    

  def train(self, new_individual, X, y, predict=False):
    start_time = time.time()
    self._reset()

    training_patterns = self._detect_training_patterns(new_individual, X, y)

    if not training_patterns:
      return False

    # Simple RBFn
    if self.metamodel_type == 'simple_rbf':
      self._determine_cluster_centers()
      self._compute_radii()

      try:
        self._compute_rbf_weights()
      except ValueError as e:
        print(f"Error during RBFN training: {e}")
        return False
      
      self.cluster_centers_opt = np.copy(self.cluster_centers)
      self.radii_opt = np.copy(self.radii)
      self.weights_opt = np.copy(self.rbf.weights)
      self._set_opt_rbf_parameters()
      
      if predict:
        return self.predict(new_individual)

      print(f"Training completed in {time.time() - start_time:.2f} seconds.")
      return True

    # Growing SOM RBFn
    total_errors = np.array([])
    maximum_errors = np.array([])
    clusters = np.array([])
    
    best_total_error = np.inf
    retries = self.max_retries

    termination_condition = False
    while not termination_condition:
      if self.cluster_centers is not None and \
        len(self.cluster_centers) > self.minmax_num_clusters[1]:
        self._set_opt_rbf_parameters()
        break
      
      self._determine_cluster_centers()
      self._compute_radii()

      try:
        self._compute_rbf_weights()
      except ValueError as e:
        print(f"Error during RBFN training: {e}")
        return False

      total_error, maximum_error, most_frequent_center = self._test()
      total_errors = np.append(total_errors, total_error)
      maximum_errors = np.append(maximum_errors, maximum_error)
      clusters = np.append(clusters, len(self.cluster_centers))

      if total_error < best_total_error:
        best_total_error = total_error
        retries = self.max_retries

        self.cluster_centers_opt = np.copy(self.cluster_centers)
        self.radii_opt = np.copy(self.radii)
        self.weights_opt = np.copy(self.rbf.weights)

      else:
        retries -= 1
        if retries == 0:
          self._set_opt_rbf_parameters()
          termination_condition = True
      
      # self._plot_convergence(clusters, total_errors, maximum_errors)
      self._split_cluster_center(most_frequent_center)

    if predict:
      return self.predict(new_individual)

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return True

  def predict(self, X):
    if self.cluster_centers_opt is None or self.radii_opt is None or self.weights_opt is None:
      raise ValueError("GrowingSomRBFN: Model is not trained yet.")
    
    if X.ndim == 1:
      return self.rbf.predict(X.reshape(1, -1))[0]

    elif X.ndim != 2:
      raise ValueError("Input X must be a 1D or 2D array.")
    
    return self.rbf.predict(X)

  def _set_opt_rbf_parameters(self):
    self.rbf.set_parameters(self.cluster_centers_opt, self.radii_opt, self.weights_opt)

  def _detect_training_patterns(self, new_individual, X, y):
    X_augmented = np.vstack([new_individual, X])
    self.mst.compute(X_augmented)
    traversed_points_indices, is_outlying = self.mst.traverse(0, None, 3, 4)
    
    if is_outlying:
      return False
    
    indices = np.array(traversed_points_indices) - 1
    
    training_patterns_x = X[indices]
    training_patterns_y = y[indices]

    self.X_train, self.X_test, self.y_train, self.y_test = \
      train_test_split(training_patterns_x, training_patterns_y, 
                       test_size=self.test_size, random_state=42)

    self.training_patterns = training_patterns_x

    return True

  def _determine_cluster_centers(self):
    if self.metamodel_type == 'growing_rbf':
      self.som.train(self.training_patterns, self.cluster_centers)
      self.cluster_centers = self.som.get_cluster_centers()
    else:
      self.cluster_centers = np.copy(self.training_patterns)

  def _compute_radii(self):
    self.radii_calculator.compute(self.cluster_centers)
    self.radii = self.radii_calculator.get_radii()

  def _compute_rbf_weights(self):
    self.rbf.set_centers_and_radii(self.cluster_centers, self.radii)
    self.rbf.train(self.X_train, self.y_train)

  def _test(self):
    X_combined = np.vstack([self.X_test, self.X_train])
    y_combined = np.concatenate([self.y_test, self.y_train])
    
    predictions, errors, total_error = self.rbf.test(X_combined, y_combined)
    # predictions, errors, total_error = self.rbf.test(self.X_test, self.y_test)
    
    closest_center_indices = np.array([
      np.argmin(np.linalg.norm(self.cluster_centers - point, axis=1))
      for point in X_combined])
    
    df = pd.DataFrame({
      'y_exact': y_combined,
      'y_predicted': predictions,
      'error': errors,
      'closest_center_index': closest_center_indices
    })
    
    sample_percentage = 0.3
    sample_count = int(len(df.index) * sample_percentage)
    
    df_sorted = df.sort_values(by='error', ascending=False).head(sample_count)
    most_frequent_center = df_sorted['closest_center_index'].value_counts().idxmax()
    maximum_error = df_sorted.iloc[0]['error']
    return total_error, maximum_error, most_frequent_center

  def _split_cluster_center(self, index):
    if self.cluster_centers is None:
      raise ValueError("Cluster centers have not been set.")
    
    if index < 0 or index >= len(self.cluster_centers):
      raise IndexError("Cluster index out of range.")

    self.cluster_centers = np.insert(
      self.cluster_centers, index, self.cluster_centers[index], axis=0)
    self.som.set_cluster_centers(self.cluster_centers)

  @staticmethod
  def _plot_convergence(clusters, total_errors, maximum_errors):
    plt.plot(clusters, total_errors, label="Total Error", marker='o')
    # plt.plot(clusters, maximum_errors, label="Maximum Errors", marker='x')
    
    plt.yscale('log')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Error (log scale)")
    plt.title("Error History")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig("error_history.png")
    plt.close()

  def _reset(self):
    self.cluster_centers = None
    self.radii = None
    self.training_patterns = None
    self.X_train, self.X_test, \
    self.y_train, self.y_test = None, None, None, None
    
    self.cluster_centers_opt = None
    self.radii_opt = None
    self.weights_opt = None
    
    if self.metamodel_type == 'growing_rbf':
      self.som.set_num_clusters(self.minmax_num_clusters[0])
