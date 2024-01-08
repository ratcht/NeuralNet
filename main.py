from autodiff_numpy import Numpy
import numpy as _np
import math
from computation_graph import ComputationGraph, Node


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



def func1(x):
  a1 = np.multiply(np.multiply(x, x),3)
  a2 = np.multiply(x, 4)
  a3 = 2
  y = np.add(np.add(a1, a2),a3)
  return y

y = func1(3)
print(str(cg))