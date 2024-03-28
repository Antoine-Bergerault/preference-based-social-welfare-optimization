from .base import Strategy

from functools import cached_property
import numpy as np

class Allocation(Strategy):
    def __init__(self, type, number_of_resources) -> None:
        super().__init__()
        
        self.type = type
        self.number_of_resources = number_of_resources
        
    @cached_property
    def dimensions(self):
        self.check_initialization()
        
        if self.actions_dimensions % self.number_of_resources != 0:
            raise ValueError("Cannot define allocation utilities where agents do not optimize the same number of resources")
        
        return self.number_of_resources
    
    def convert(self, x):
        self.check_initialization()
        
        assert len(x) == self.dimensions
        
        def f(t):
            if not isinstance(t, np.ndarray):
                t = np.array([t]).squeeze()

            # first axis is optional
            # need to contain at least one point of the correct shape
            if t.ndim < 2:
                t = t[None, ...]
            
            assert t.shape[-1] == self.actions_dimensions

            t = t.reshape((-1, self.actions_dimensions // self.number_of_resources, self.number_of_resources))
            
            # convert every player action to proportion on resources
            t = np.max(t, 0, keepdims=True) # discard negative values
            t /= np.sum(t, axis=-1)+1e-5
            
            if self.type == "linear":
                # each resource is worth the same for each player
                # the optimum is then for each to play greedy
                return np.einsum("ijk,k->i", t, x)
            elif self.type == "submodular":
                units_per_resource = np.einsum("ijk,k->ik", t, x)
                # TODO: make submodular function configurable
                values = np.log(units_per_resource * (np.e-1) + 1) # roughly divide the return by 2 for each new unit
                return np.sum(values, axis=1)
            
            raise ValueError(f"Unknown type \"{self.type}\" for utilities allocation strategy")
            
        return f