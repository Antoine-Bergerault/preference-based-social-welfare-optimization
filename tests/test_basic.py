from pref import pref_social_welfare
from pref.utils import (
    retrieve_learning_algorithm, 
    retrieve_reward_type, 
    retrieve_social_welfare_function,
    define_social_welfare_function
)

from utils.dummy import *

def test_dummy_available():
    retrieve_learning_algorithm("dummy")
    retrieve_reward_type("dummy")
    retrieve_social_welfare_function("dummy")
    
def test_algorithm_works_with_all_dummy():
    pref_social_welfare({
        "reward_type": "dummy",
        "social_welfare_function": "dummy", 
        "learning_algorithm": "dummy"
    })

def test_dimensions_social_welfare_function():
    def assert_shape_wf(x, dim):
        assert x.shape == (dim, )
        return 0
        
    for n_dimensions in [1, 2, 4]:
        tag = f"sw-test-dim-{n_dimensions}"
        
        decorator = define_social_welfare_function(tag, input_dim=n_dimensions)
        decorator(lambda x: assert_shape_wf(x, n_dimensions))
        
        pref_social_welfare({
            "reward_type": "dummy",
            "social_welfare_function": tag, 
            "learning_algorithm": "dummy",
            "horizon": 3
        })