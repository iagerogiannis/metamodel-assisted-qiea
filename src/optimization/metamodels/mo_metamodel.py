import numpy as np
from .growing_som_rbfn import GrowingSOMRBFNetwork
import multiprocessing

class MOMetamodel:
  def __init__(self, config, num_of_objectives=1):
    metamodel_type = config['metamodel_type']
    minmax_num_clusters = config['som']['num_clusters']
    test_size = config['rbfn']['test_size']
    max_retries = config['termination_criteria']['max_retries']
    if num_of_objectives > 1:
      self.gsomrbfn = [
        GrowingSOMRBFNetwork(minmax_num_clusters, test_size, max_retries, metamodel_type)
        for _ in range(num_of_objectives)
      ]
      self.multi_objective = True
    else:
      self.gsomrbfn = GrowingSOMRBFNetwork(minmax_num_clusters, test_size, max_retries, metamodel_type)
      self.multi_objective = False

  def predict(self, new_individual, X, y):
    y = np.asarray(y)
    if y.ndim == 1 or not self.multi_objective:
      return self.gsomrbfn.train(new_individual, X, y, predict=True)

    # Multi-objective: use multiprocessing
    manager = multiprocessing.Manager()
    results = manager.list([None] * y.shape[1])
    processes = []

    def worker(i, results):
      results[i] = self.gsomrbfn[i].train(new_individual, X, y[:, i], predict=True)

    for i in range(y.shape[1]):
      p = multiprocessing.Process(target=worker, args=(i, results))
      processes.append(p)
      p.start()

    for p in processes:
      p.join()

    return list(results)
