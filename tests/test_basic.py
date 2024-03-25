from pref import pref_social_welfare
from pref.utils import retrieve_learning_algorithm, retrieve_reward_type, retrieve_social_welfare_function

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