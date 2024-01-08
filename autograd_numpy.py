from computation_graph import ComputationGraph
from numpy_wrapping import wrap_namespace, func_mapping

class Numpy:
  def __init__(self, cg: ComputationGraph=None):
    if cg == None: 
      cg = ComputationGraph()
    
    self.cg = cg

    # Bind numpy functions
    wrap_namespace(func_mapping, self, cg)
