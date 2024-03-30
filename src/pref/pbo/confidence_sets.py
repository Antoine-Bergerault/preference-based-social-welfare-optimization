import numpy as np
from math import log, exp

from GPy.kern import Linear, Exponential, Matern32, RBF

class Ball():
    def __init__(self, kernel, kernel_lengthscale, bound, beta, input_dim, hist_data=None, pref_data=None) -> None:
        assert kernel in ["linear", "exponential", "gaussian", "matern"], f"Unknown kernel type \"{kernel}\""
        
        self._kernel = kernel
        self.kernel_lengthscale = kernel_lengthscale
        
        # note that the parameters are fixed for all the kernels
        if kernel == "linear":
            self._kernel_fun = Linear(input_dim) # no lengthscale at use here
        elif kernel == "exponential":
            self._kernel_fun = Exponential(input_dim, lengthscale=kernel_lengthscale)
        elif kernel == "gaussian":
            self._kernel_fun = RBF(input_dim, lengthscale=kernel_lengthscale)
        elif kernel == "matern":
            self._kernel_fun = Matern32(input_dim, lengthscale=kernel_lengthscale)
        
        self.bound = bound
        self.beta = beta
        self.input_dim = input_dim
        self.hist_data = hist_data
        self.pref_data = pref_data
    
    def kernel(self, x_1, x_2):
        return self._kernel_fun.K(x_1, x_2)
    
    def initial(self, x):
        assert self.hist_data is None
        self.hist_data = np.array([x])
    
    def update(self, actions, preference):
        assert self.hist_data is not None and len(self.hist_data) > 0
        new_hist_data = np.vstack((self.hist_data, np.array(actions)))
        new_pref_data = np.array(preference) if self.pref_data is None else np.vstack((self.pref_data, np.array(preference)))
        
        return Ball(self._kernel, self.kernel_lengthscale, self.bound, self.beta, self.input_dim, new_hist_data, new_pref_data)