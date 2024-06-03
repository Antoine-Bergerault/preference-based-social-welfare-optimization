import numpy as np

from pref.utils import define_learning_algorithm, maximize

@define_learning_algorithm("oracle")
def maximizer_oracle(function, input_dim):
    # TODO: change this fixed range. Ideally the search range should
    # depend on the function and the dimension
    search_range = np.arange(-12, 12, 0.3)
    
    # if a list is passed, we assume that all the functions in the list are the same
    # thus we only maximize one (maximizing the sum will require more computation)
    if isinstance(function, list):
        function = function[0]
    
    return maximize(function, input_dim, search_range)