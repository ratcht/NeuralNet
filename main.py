import numpy as np
import math




def grad_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):  
  vector = start
  for _ in range(n_iter):
    diff = -learn_rate * gradient(vector)
    if np.all(np.abs(diff) <= tolerance):
      break
    vector += diff
  return vector

def squared_loss(y, y_hat):
  return math.pow(abs(y-y_hat), 2)



def relu(z):
  return max(0, z)

def sigmoid(z):
  return 1/(1+math.exp(-z))

def softplus(z):
  return math.log(1+math.exp(z))

def tanh(z):
  return (math.exp(2*z)-1)/(math.exp(2*z)+1)

activation_func_map = {"relu": relu, "sigmoid": sigmoid, "softplus": softplus, "tanh": tanh}


class Unit:
  def __init__(self, weights: np.array, activation):
    self.weights = weights
    self.activation = activation
  
  def update_weights(self, weights: np.array):
    self.weights = weights

  def res(self, inputs):
    a = self.activation(np.dot(self.weights, inputs))
    return a
  

class DenseLayer:
  def __init__(self, num_inputs, num_outputs, activation: str):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs

    self.activation = activation_func_map.get(activation)
    self.units: np.array[Unit] = np.zeros([num_outputs], dtype=Unit)

    for i in range(0, num_outputs):
      self.units[i] = Unit(weights=np.zeros(num_inputs+1), activation=self.activation)
    
  def forward(self, inputs: np.array):
    outputs = np.zeros([self.num_outputs])
    for i in range(0, self.num_outputs):
      # insert 1 to inputs
      inputs_dummy = np.insert(inputs, 0, 1)

      outputs[i] = self.units[i].res(inputs_dummy)

    return outputs
  

class Network:
  def __init__(self, input_layer: np.array):
    self.input_layer = input_layer
    self.layers = []



dense = DenseLayer(2, 1, activation="relu")
print(dense.forward(np.array([1,2])))