import math


def relu(z):
  return max(0, z)

def sigmoid(z):
  return 1/(1+math.exp(-z))

def softplus(z):
  return math.log(1+math.exp(z))

def tanh(z):
  return (math.exp(2*z)-1)/(math.exp(2*z)+1)



activation_func_map = {"relu": relu, "sigmoid": sigmoid, "softplus": softplus, "tanh": tanh}