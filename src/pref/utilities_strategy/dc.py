from .base import Strategy

from functools import cached_property
import numpy as np

class DCConverter(Strategy):
    def __init__(self, precision, period, kernel_type) -> None:
        super().__init__()
        
        self.precision = precision
        self.period = period
        self.kernel_type = kernel_type

    @cached_property
    def kernel(self):
        if self.kernel_type == "sinc":
            return np.sinc
        elif self.kernel_type == "gaussian":
            return lambda t: np.exp(-t**2/2) # leading factor incorporated in the coefficient
        raise ValueError(f"\"{self.kernel_type}\" not recognized as a valid kernel type for D/C converter")

    @cached_property
    def dimensions(self):
        self.check_initialization()
        return self.precision ** self.actions_dimensions
    
    def convert(self, x):
        self.check_initialization()
        
        assert len(x) == self.dimensions
        
        coefficients = x.reshape((self.precision,) * self.actions_dimensions)
        
        indices = np.arange(self.precision)
        
        # Reference/Knot points
        ref_points = self.period * np.stack(np.meshgrid(
            *np.repeat([indices], self.actions_dimensions, axis=0)
        ), axis=-1)
        
        dist = lambda t: np.linalg.norm(t[:, *[None] * self.actions_dimensions, ...] - ref_points[None, ...], axis=-1)
        
        kernel = self.kernel
        base = lambda t: kernel(dist(t) / self.period)
        
        def f(t):
            if not isinstance(t, np.ndarray):
                t = np.array([t]).squeeze()

            # first axis is optional
            # need to contain at least one point of the correct shape
            if t.ndim < 2:
                t = t[None, ...]
            
            assert t.shape[-1] == self.actions_dimensions

            return np.sum((base(t) * coefficients).reshape((len(t), -1)), axis=1)

        return f