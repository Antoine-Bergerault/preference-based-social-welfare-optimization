import numpy as np

from pref.utils import define_learning_algorithm, maximize

@define_learning_algorithm("oracle")
def maximizer_oracle(function, input_dim):
    # TODO: change this fixed range. Ideally the search range should
    # depend on the function and the dimension
    search_range = np.arange(-16, 16, 0.5)
    
    return maximize(function, input_dim, search_range)