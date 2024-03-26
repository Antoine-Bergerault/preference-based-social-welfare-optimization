import numpy as np
from math import log, exp

from GPy.kern import Linear, Exponential, Matern32

class Ball():
    def __init__(self, kernel, bound, epsilon, delta, input_dim, hist_data=None, pref_data=None) -> None:
        assert kernel in ["linear", "exponential", "matern"], f"Unknown kernel type \"{kernel}\""
        
        self._kernel = kernel
        
        # note that the parameters are fixed for all the kernels
        if kernel == "linear":
            self._kernel_fun = Linear(input_dim)
        elif kernel == "exponential":
            self._kernel_fun = Exponential(input_dim)
        elif kernel == "matern":
            self._kernel_fun = Matern32(input_dim)
            
        self.bound = bound
        self.epsilon = epsilon
        self.delta = delta
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
        
        return Ball(self._kernel, self.bound, self.epsilon, self.delta, self.input_dim, new_hist_data, new_pref_data)
    
    def covering_number(self, epsilon):
        # use approximations found in J (proof of theorem 11)
        if self._kernel == "linear":
            return 1/epsilon
        elif self._kernel == "exponential":
            return 1/epsilon # TODO: Change this with an accurate approximation for the exponential kernel
        elif self._kernel == "matern":
            return exp(-(1/epsilon)**(3/2)*log(epsilon))