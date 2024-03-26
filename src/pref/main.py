from typing import Callable, Dict, Union, Tuple
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver('div', lambda x, y: x/y)

import numpy as np
import numpy.typing as npt

from pref.utils import (
    sigmoid,
    retrieve_reward_type, 
    retrieve_learning_algorithm,
    retrieve_social_welfare_function
)

from pref.pbo.base import ucb_function
from pref.pbo.confidence_sets import Ball

Action = npt.NDArray[float]
UCB = Callable[[Action], float]
Reward = Callable[[Action], float]

RewardFunction = Callable[UCB, Callable[Action, Reward]]
LearningAlgorithm = Callable[[RewardFunction, int], Tuple[Action, Reward]]

def pref_social_welfare_generator(config: Union[DictConfig, Dict]):
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    sw = retrieve_social_welfare_function(config.get("social_welfare_function", "beale"))
    social_welfare, input_dim = sw.fun, sw.input_dim
    
    assert input_dim is not None
    
    initial_actions = np.array([0] * input_dim)
    
    # define horizon (maximum number of turns)
    T = config.get("horizon", 30)

    confidence_set = Ball(
        kernel=config.get("kernel_type", "linear"),
        bound=config.get("RKHS_bound", 0.2),
        epsilon=config.get("epsilon", 1/T),
        delta=config.get("delta", 0.02),
        input_dim=input_dim
    )
    confidence_set.initial(initial_actions)
    
    # quick fix for no initial data
    # TODO: Make sure to do proper initialization instead
    confidence_set = confidence_set.update(np.array([-3.4] * input_dim), 1)
    
    reward_type: RewardFunction = retrieve_reward_type(config.get("reward_type", "tw"))
    learning_algorithm: LearningAlgorithm = retrieve_learning_algorithm(config.get("learning_algorithm", "hedge"))

    last_actions = initial_actions
    for t in range(T):
        # find next point to query
        
        s_ucb = ucb_function(confidence_set)
        
        # define rewards from ucb
        
        reward_function = reward_type(s_ucb)
        
        # run the rational agents
        
        actions, rewards = learning_algorithm(reward_function, input_dim)
        
        # measure preference
        
        probability = sigmoid(social_welfare(actions) - social_welfare(last_actions))
        preference = np.random.binomial(1, probability)
        
        confidence_set = confidence_set.update(actions, preference)
        
        last_actions = actions
        
        yield (s_ucb, actions, rewards, preference)

def pref_social_welfare(config: Union[DictConfig, Dict], generator=False):
    generator_object = pref_social_welfare_generator(config)
    
    if generator:
        return generator_object
    
    for _ in generator_object:
        pass