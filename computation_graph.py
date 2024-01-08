import autograd_numpy as _np
import math
from contextlib import contextmanager
from typing import Callable
from functools import wraps




class TraceStack():
  def __init__(self):
    self.top = -1

  def new_trace(self):
    """Increment trace depth."""
    self.top += 1
    return self.top




class Node:
  def __init__(self, val, fun, parents, keep_grad, *fun_args, **fun_kwargs):
    self.val = val
    self.fun = fun
    self.parents = parents
    self.trace_id = -1
    self.keep_grad = keep_grad

  def __repr__(self): 
    """A (very) basic string representation"""
    if self.val is None: 
      str_val = 'None'
    else: 
      str_val = str(round(self.val,3))
    
    parents_ids = [node.trace_id for node in self.parents]

    repr = f"Node ID: {self.trace_id}\nFun: {str(self.fun)}\nValue: {str_val}\nParents: {str(parents_ids)}"

    return  repr
  
  def set_trace_id(self, i):
    self.trace_id = i
  
  @classmethod
  def start_node(cls, value=None, keep_grad=True):
    fun = lambda x: x
    parents = []
    return cls(value, fun, parents, keep_grad)

class ComputationGraph:
  def __init__(self):
    self.trace_stack = TraceStack()
    self.graph = []

  def add_to_graph(self, x: Node):
    id = self.trace_stack.new_trace()
    x.set_trace_id(id)
    self.graph.append(x)

  def __repr__(self):
    repr = ""
    for node in self.graph:
      repr += str(node)
      repr += "\n\n"
    
    return repr




