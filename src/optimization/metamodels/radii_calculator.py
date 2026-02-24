import numpy as np
from .mst import MinimumSpanningTree


class RadiiCalculator:
  def __init__(self, max_num_branches=4):
    self.mst = MinimumSpanningTree()
    self.max_num_branches = max_num_branches
    self.centers = None
    self.radii = None

  def compute(self, centers: np.ndarray = None):
    if centers is not None:
      self.set_centers(centers)
    
    for m in range(len(self.centers)):
      traversed_center_indices = self.mst.traverse(m,
                                                   min_num_branches=self.max_num_branches,
                                                   max_num_branches=self.max_num_branches)
      self.radii[m] = np.sum([np.linalg.norm(self.centers[m] - self.centers[i]) for i in traversed_center_indices]) \
        / len(traversed_center_indices)

  def set_centers(self, centers: np.ndarray):
    self.centers = centers
    self.mst.compute(centers)
    self.radii = np.zeros(len(centers))

  def get_radii(self):
    return self.radii
