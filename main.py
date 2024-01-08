from autograd_numpy import Numpy
import numpy as _np
import math
from contextlib import contextmanager
from typing import Callable

from functools import wraps
from computation_graph import ComputationGraph


# init numpy with a cg
cg = ComputationGraph()
np = Numpy(cg)


def grad_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):  
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * gradient(vector)
    if _np.all(_np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector

def squared_loss(y, y_hat):
  return math.pow(abs(y-y_hat), 2)




def test(x, y):
  z =  np.add(x,y) 
  y = np.multiply(z, 5)
  return y

m = test(5, 7)
print("\n")
print(str(cg))