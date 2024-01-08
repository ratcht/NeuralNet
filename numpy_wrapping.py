import numpy as _np
import types
from functools import wraps
from computation_graph import ComputationGraph, Node

func_mapping = {
  "add": _np.add,
  "subtract": _np.subtract,
  "multiply": _np.multiply,
  "divide": _np.divide,
  "sin": _np.sin,
  "cos": _np.cos
  }


def primitive(f, cg: ComputationGraph, keep_grad=True): 
  @wraps(f)
  def inner(*args, **kwargs): 
    """This is a nested function"""
    # add to graph


    ## Code to add operation/primitive to computation graph
    # We need to separate out the integer/non node case. Sometimes you are adding 
    # constants to nodes. 
    def getval(o):
      return o.val if type(o) == Node else o
    
    if len(args):
      argvals = [getval(o) for o in args]
    else:
      argvals = args
    if len(kwargs):
      kwargvals = dict([(k,getval(o)) for k,o in kwargs.items()])
    else:
      kwargvals =  kwargs
      
    # get parents 
    l = list(args) + list(kwargs.values())
    parents = [o for o in l if type(o) == Node ]
    
    value = f(*argvals, **kwargvals)
    print("add", "'" + f.__name__ + "'", "to graph with value",value)
    
    node = Node(value, f, parents, keep_grad)

    cg.add_to_graph(node)

    return node
  return inner


def wrap_namespace(old: dict, cls, cg: ComputationGraph):
  """Performs triage on objects from numpy, copying them from old to new namespace. 
      old: __dict__ from original numpy
      new: dict to copy old into 
      """
  # Taken from here: 
  # https://github.com/mattjj/autodidact/blob/b3b6e0c16863e6c7750b0fc067076c51f34fe271/autograd/numpy/numpy_wrapper.py#L8 
  nograd_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
    _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
    _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
    _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
    _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
    _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
    _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
    _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
    _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
    _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
    _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
  ]
  function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}

  for name,obj in old.items(): 
    if obj in nograd_functions:  
      # non-differentiable functions
      #new[name] = primitive(obj, keep_grad=False)

      setattr(cls, name, primitive(obj, cg, keep_grad=False))

    elif type(obj) in function_types:  # functions with gradients 
      # differentiable functions
      #new[name] = primitive(obj)

      setattr(cls, name, primitive(obj, cg))

    else: 
      # just copy over 
      #new[name] = obj

      setattr(cls, name, obj)

