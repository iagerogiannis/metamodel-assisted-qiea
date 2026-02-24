import numpy as np


def rbf_gaussian(x, center, radius):
  """
  Gaussian RBF function
  This function computes the Gaussian RBF value for a given input x,
  center, and radius.
  """
  return np.exp(-((x - center) ** 2) / (2 * (radius ** 2)))


class RadialBasisFunctionNetwork:
  """
  Radial Basis Function (RBF) class
  This class represents a Radial Basis Function (RBF) network.
  It initializes with centers and radii, and provides a method to train the network.
  The training method uses the least squares solution to find the weights.
  """

  def __init__(self, interpolate=False):
    self.interpolate = interpolate
    self.centers = None
    self.radii = None
    self.weights = None

  def _calculate_design_matrix(self, X):
    """
    Calculate the design matrix G
    This method computes the design matrix G using the RBF function.
    """
    if X.ndim != 2:
      raise ValueError("Input X must be a 2D array.")
    
    G = np.zeros((X.shape[0], len(self.centers)))
    
    for i, center in enumerate(self.centers):
      distances = np.linalg.norm(X - center, axis=1)
      G[:, i] = rbf_gaussian(distances, 0, self.radii[i])
    
    return G

  def _calculate_errors(self, G, y):
    """
    Calculate the errors
    This method computes the total error and node errors.
    """
    if G.ndim != 2:
      raise ValueError("Input G must be a 2D array.")
    
    if y.ndim != 1:
      raise ValueError("Output y must be a 1D array.")
    
    if G.shape[0] != y.shape[0]:
      raise ValueError("Number of rows in G must match the length of y.")
    
    if np.any(np.isnan(G)) or np.any(np.isinf(G)):
      raise ValueError("Design matrix G contains invalid values (NaN or Inf).")
    
    predictions = G @ self.weights
    individual_errors = np.abs((predictions - y) / y)
    diff = y - predictions
    diff = np.clip(diff, -1e10, 1e10)  # Prevent overflow in square
    total_error = 0.5 * np.sum(np.square(diff))
    
    return predictions, individual_errors, total_error

  def _solve(self, X, y, train=False):
    """
    Train the RBF network
    This method computes the design matrix G using the RBF function
    and solves for the weights using the least squares method.
    """
    if self.centers is None or self.radii is None:
      raise ValueError("RBFN: Centers or radii have not been set." + 
        " Use set_centers_and_radii() to set them.")

    if X.ndim != 2:
      raise ValueError("Input X must be a 2D array.")
    
    if y.ndim != 1:
      raise ValueError("Output y must be a 1D array.")
    
    if X.shape[0] != y.shape[0]:
      raise ValueError("Number of rows in X must match the length of y.")
    
    G = self._calculate_design_matrix(X)

    if train:
      weights = None

      # Interpolation: solve G @ weights = y
      if self.interpolate:
        try:
          weights = np.linalg.solve(G, y)
        except np.linalg.LinAlgError:
          weights = np.linalg.lstsq(G, y, rcond=None)[0]

      # Approximation: solve G^T @ G @ weights = G^T @ y
      else:
        try:
          weights = np.linalg.solve(G.T @ G, G.T @ y)
        except np.linalg.LinAlgError:
          weights = np.linalg.lstsq(G, y, rcond=None)[0]
        
      if weights is None:
        raise ValueError("Model weights are not properly initialized.")
      
      if np.any(np.isnan(weights)):
        raise ValueError(f"Model weights contain NaN values: {weights}")
      
      if np.any(np.isinf(weights)):
        raise ValueError(f"Model weights contain Inf values: {weights}")
      
      self.weights = weights
    
    return self._calculate_errors(G, y)
  
  def train(self, X, y):
    predictions, individual_errors, total_error = self._solve(X, y, True)
    return self.weights, predictions, individual_errors, total_error

  def test(self, X, y):
    try:
      return self._solve(X, y, False)
    except ValueError as e:
      print(f"Error during testing: {e}")
      return None, None, None

  def predict(self, X):
    """
    Predict using the RBF network
    This method computes the output of the RBF network for a given input X.
    """
    if self.weights is None:
      raise ValueError("The model has not been trained yet.")
    
    if X.ndim != 2:
      raise ValueError("Input X must be a 2D array.")
    
    G = self._calculate_design_matrix(X)
    prediction = G @ self.weights
    return prediction

  def set_centers_and_radii(self, centers, radii):
    if len(centers) != len(radii):
      raise ValueError("Centers and radii must have the same length.")
    
    if centers.ndim != 2:
      raise ValueError("Centers must be a 2D array.")
    
    if radii.ndim != 1:
      raise ValueError("Radii must be a 1D array.")
    
    self.centers = centers
    self.radii = radii

  def set_parameters(self, centers, radii, weights):
    if len(weights) != len(centers):
      raise ValueError("Weights and centers must have the same length.")
    
    self.set_centers_and_radii(centers, radii)
    self.weights = weights
