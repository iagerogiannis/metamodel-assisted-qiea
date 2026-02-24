import numpy as np


def inverse_decay_to_zero(parameter, n, max_iter):
    """Inverse decay function to zero."""
    C = max_iter / 100.0
    return parameter * C / (C + n)

def linear_decay_to_zero(parameter, n, max_iter):
    """Linear decay function to zero."""
    return parameter * (1 - n / max_iter)

def inverse_decay_to_one(parameter, n, max_iter):
    """Inverse decay function to one."""
    C = (parameter - 1) / max_iter
    return parameter / (1 + (n * C))

def linear_decay_to_one(parameter, n, max_iter):
    """Linear decay function to one."""
    return parameter + (n * (1 - parameter) / max_iter)

def asymptotic_decay(parameter, t, max_iter):
    """Asymptotic decay function."""
    return parameter / (1 + t / (max_iter / 2))

def gaussian(center_diff, sigma):
    """Gaussian neighborhood function."""
    return np.exp(-np.linalg.norm(center_diff)**2 / (2 * sigma**2))

def mexican_hat(center_diff, sigma):
    """Mexican Hat (Ricker wavelet) neighborhood function."""
    r = np.linalg.norm(center_diff)
    return (1 - r**2 / sigma**2) * np.exp(-r**2 / (2 * sigma**2))

def bubble(center_diff, sigma):
    """Bubble neighborhood function (returns 1 if within sigma, 0 otherwise)."""
    return 1.0 if np.linalg.norm(center_diff) <= sigma else 0.0

def triangle(center_diff, sigma):
    """Triangular neighborhood function (linear decay to 0)."""
    dist = np.linalg.norm(center_diff)
    return max(0.0, 1 - dist / sigma)
