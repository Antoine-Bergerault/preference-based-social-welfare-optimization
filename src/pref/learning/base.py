import itertools
import numpy as np

from pref.utils import define_learning_algorithm

@define_learning_algorithm("oracle")
def maximizer_oracle(function, input_dim):
    argmax = None
    max = None
    
    # TODO: change this fixed range. Ideally the search range should
    # depend on the function and the dimension
    search_range = np.arange(-3, 3, 0.3)
    
    for pre_image in itertools.product(*np.repeat([search_range], input_dim, axis=0)):
        reward = function(pre_image)
        if (max is None) or (reward > max):
            argmax = np.array(pre_image)
            max = reward

    return argmax, max