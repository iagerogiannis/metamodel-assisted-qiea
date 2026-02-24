import numpy as np

def transpose_2d_list(l):
  return list(map(list, zip(*l)))


def custom_zip(*lists):
  return [list(a) for a in zip(*lists)]
  

def flatten(xss):
  return [x for xs in xss for x in xs]


def average(*args):
  return sum(args) / len(args)


def list_average(l):
  if len(l) == 0:
    return 0
  
  if isinstance(l[0], (list, np.ndarray)):
    min_length = min(len(item) for item in l)
    truncated = [item[:min_length] for item in l]
    return np.mean(truncated, axis=0)
  
  return sum(l) / len(l)