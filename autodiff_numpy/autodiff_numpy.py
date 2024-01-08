from computation_graph import ComputationGraph, Node
import numpy as _np
from .numpy_wrapping import wrap_namespace

class Numpy:
  def __init__(self, cg: ComputationGraph=None):
    if cg == None: 
      cg = ComputationGraph()
    
    self.cg = cg

    # Bind numpy functions
    wrap_namespace(_np.__dict__, self, cg)

    anp = self.__dict__

    setattr(Node, 'ndim', property(lambda self: self.val.ndim))
    setattr(Node, 'size', property(lambda self: self.val.size))
    setattr(Node, 'dtype',property(lambda self: self.val.dtype))
    setattr(Node, 'T', property(lambda self: anp['transpose'](self)))
    setattr(Node, 'shape', property(lambda self: self.val.shape))

    setattr(Node,'__len__', lambda self, other: len(self._val))
    setattr(Node,'astype', lambda self,*args,**kwargs: anp['_astype'](self, *args, **kwargs))
    setattr(Node,'__neg__', lambda self: anp['negative'](self))
    setattr(Node,'__add__', lambda self, other: anp['add'](     self, other))
    setattr(Node,'__sub__', lambda self, other: anp['subtract'](self, other))
    setattr(Node,'__mul__', lambda self, other: anp['multiply'](self, other))
    setattr(Node,'__pow__', lambda self, other: anp['power'](self, other))
    setattr(Node,'__div__', lambda self, other: anp['divide'](  self, other))
    setattr(Node,'__mod__', lambda self, other: anp['mod'](     self, other))
    setattr(Node,'__truediv__', lambda self, other: anp['true_divide'](self, other))
    setattr(Node,'__matmul__', lambda self, other: anp['matmul'](self, other))
    setattr(Node,'__radd__', lambda self, other: anp['add'](     other, self))
    setattr(Node,'__rsub__', lambda self, other: anp['subtract'](other, self))
    setattr(Node,'__rmul__', lambda self, other: anp['multiply'](other, self))
    setattr(Node,'__rpow__', lambda self, other: anp['power'](   other, self))
    setattr(Node,'__rdiv__', lambda self, other: anp['divide'](  other, self))
    setattr(Node,'__rmod__', lambda self, other: anp['mod'](     other, self))
    setattr(Node,'__rtruediv__', lambda self, other: anp['true_divide'](other, self))
    setattr(Node,'__rmatmul__', lambda self, other: anp['matmul'](other, self))
    setattr(Node,'__eq__', lambda self, other: anp['equal'](self, other))
    setattr(Node,'__ne__', lambda self, other: anp['not_equal'](self, other))
    setattr(Node,'__gt__', lambda self, other: anp['greater'](self, other))
    setattr(Node,'__ge__', lambda self, other: anp['greater_equal'](self, other))
    setattr(Node,'__lt__', lambda self, other: anp['less'](self, other))
    setattr(Node,'__le__', lambda self, other: anp['less_equal'](self, other))
    setattr(Node,'__abs__', lambda self: anp['abs'](self))
    setattr(Node,'__hash__', lambda self: id(self))
