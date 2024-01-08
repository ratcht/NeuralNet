import autograd_numpy as np
import numpy as _np
from activation_funcs import activation_func_map

class Unit:
  def __init__(self, weights: _np.array, activation):
    self.weights = weights
    self.activation = activation
  
  def update_weights(self, weights: _np.array):
    self.weights = weights

  def res(self, inputs):
    a = self.activation(_np.dot(self.weights, inputs))
    return a
  

class DenseLayer:
  def __init__(self, num_inputs, num_outputs, activation: str):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs

    self.activation = activation_func_map.get(activation)
    self.units: _np.array[Unit] = _np.zeros([num_outputs], dtype=Unit)

    for i in range(0, num_outputs):
      self.units[i] = Unit(weights=_np.zeros(num_inputs+1), activation=self.activation)
    
  def forward(self, inputs: _np.array):
    outputs = _np.zeros([self.num_outputs])
    for i in range(0, self.num_outputs):
      # insert 1 to inputs
      inputs_dummy = _np.insert(inputs, 0, 1)

      outputs[i] = self.units[i].res(inputs_dummy)

    return outputs
  

class Network:
  def __init__(self, input_layer: _np.array):
    self.input_layer = input_layer
    self.layers = []