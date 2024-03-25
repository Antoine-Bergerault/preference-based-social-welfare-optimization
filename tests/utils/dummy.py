import numpy as np

from pref.utils import (
    define_social_welfare_function, 
    define_learning_algorithm, 
    define_reward_type
)

@define_reward_type("dummy")
def dummy_reward_type(ucb):
    def f(*args, **kwargs):
        return 0
    return f

@define_social_welfare_function("dummy", input_dim=2)
def dummy_social_welfare_function(x):
    return 0

@define_learning_algorithm("dummy")
def dummy_learning_algorithm(function, input_dim):
    return np.array([0]*input_dim), 0