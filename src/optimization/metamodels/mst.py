import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

class MinimumSpanningTree:
    def __init__(self):
        self.points = None
        self.dim = None
        self.num_points = None
        self.mst = None
        self.mst_branch_length_mean = None
        self.mst_branch_length_stdev = None

    def compute(self, points:np.ndarray = None):
        if points is not None:
            self.set_points(points)

        if self.points is None:
            raise ValueError("Points have not been set. Call set_points() first or pass points to compute().")

        coords = np.stack(self.points)
        dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        self.mst = minimum_spanning_tree(csr_array(dist_matrix))
        
        self.mst_branch_length_mean = np.average(self.mst.data)
        self.mst_branch_length_stdev = np.std(self.mst.data)

        return self.mst

    def traverse(self, start_index=0, max_num_branches=None, branch_length_tolerance=None, min_num_branches=None):
      if self.mst is None:
        raise ValueError("MST has not been computed yet. Call compute() first.")

      mst = self.mst
      points = self.points
      num_points = self.num_points

      # Precompute neighbors for all points for fast lookup
      mst_csr = mst.tocsr()
      mst_csc = mst.tocsc()

      def get_neighbors(idx, exclude=None):
        # Get all neighbors of idx, excluding 'exclude'
        rows = mst_csr[idx].nonzero()[1]
        cols = mst_csc[:, idx].nonzero()[0]
        neighbors = set(rows).union(cols)
        if exclude is not None:
          neighbors.discard(exclude)
        return neighbors

      traversed_points = []
      traversed_branch_lengths = []
      traversed_branch_lengths_mean = None
      traversed_branch_lengths_stdev = None
      neighboring_points = []

      # Use a set for fast membership checks
      visited = set()
      visited.add(start_index)

      # Initialize with neighbors of start_index
      for n in get_neighbors(start_index):
        branch_length = mst[start_index, n] if mst[start_index, n] != 0 else mst[n, start_index]
        neighboring_points.append((n, branch_length, start_index))

      def satisfies_branch_length_tolerance(branch_length):
        if branch_length_tolerance is None:
          return True
        if min_num_branches is not None and len(traversed_points) < min_num_branches:
          return True
        if traversed_branch_lengths_mean is None or traversed_branch_lengths_stdev == 0:
          return True
        return branch_length <= traversed_branch_lengths_mean + branch_length_tolerance * traversed_branch_lengths_stdev

      while neighboring_points:
        if max_num_branches is not None and len(traversed_points) >= max_num_branches:
          break

        # Pop the neighbor with the smallest branch length
        neighboring_points.sort(key=lambda x: x[1])
        point, branch_length, parent = neighboring_points.pop(0)
        if point in visited:
          continue
        if not satisfies_branch_length_tolerance(branch_length):
          continue

        traversed_points.append(point)
        traversed_branch_lengths.append(branch_length)
        visited.add(point)

        # Update stats
        traversed_branch_lengths_mean = np.mean(traversed_branch_lengths)
        traversed_branch_lengths_stdev = np.std(traversed_branch_lengths)

        # Add new neighbors
        for n in get_neighbors(point, exclude=parent):
          if n not in visited:
            bl = mst[point, n] if mst[point, n] != 0 else mst[n, point]
            neighboring_points.append((n, bl, point))

      # Outlier detection: compare distances from start_index to first traversed points
      def is_outlying(num_test_neighbors=2, tolerance=3):
        if not traversed_points or traversed_branch_lengths_mean is None or traversed_branch_lengths_stdev == 0:
          return False
        for k in range(min(num_test_neighbors, len(traversed_points))):
          distance = np.linalg.norm(points[start_index] - points[traversed_points[k]])
          if distance > traversed_branch_lengths_mean + tolerance * traversed_branch_lengths_stdev:
            return True
        return False

      return traversed_points, is_outlying()

    def plot(self, filename, new_individual=None, traversed_points=None):
        if self.dim not in [2, 3]:
            return

        coords = np.stack(self.points)
        is_3d = self.dim == 3

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if is_3d else plt
        
        rest_points = set(range(self.num_points)) - set(traversed_points or []) - set([new_individual])
        if rest_points:
            rest_coords = coords[list(rest_points)]
            ax.scatter(
                *rest_coords.T, 
                c='black', marker='o', label=f'{len(self.points)} Closest'
            )

        if new_individual is not None:
            ax.scatter(
                *coords[new_individual], 
                c='red', marker='^', label='Prediction Point'
            )

        if traversed_points is not None:
            traversed_coords = coords[traversed_points]
            ax.scatter(
                *traversed_coords.T, 
                c='blue', marker='s', label='Selected'
            )

        if self.mst is not None:
            for i in range(self.num_points):
                for j in range(self.num_points):
                    if self.mst[i, j] != 0:
                        plot_style = 'b-' \
                                if traversed_points \
                                    and i in traversed_points + [new_individual] \
                                    and j in traversed_points + [new_individual] \
                                else 'k--'
                        ax.plot(
                            *zip(coords[i], coords[j]), 
                            plot_style
                        )

        for idx, point in enumerate(coords):
            ax.text(*point, str(idx), fontsize=8, ha='right')

        if new_individual is not None or len(traversed_points or []) > 0:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.175), ncol=3)

        plt.title("Minimum Spanning Tree")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def set_points(self, points: np.ndarray):
        if points.ndim != 2:
          raise ValueError("Points must be a 2D array.")

        self.points = points
        
        lengths = {len(p) for p in self.points}
        if len(lengths) != 1:
            raise ValueError("All points must have the same dimensionality.")
        
        self.dim = len(self.points[0])
        self.num_points = len(self.points)
