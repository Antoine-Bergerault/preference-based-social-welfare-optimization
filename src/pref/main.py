from typing import Callable, Dict, Union, Tuple
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import numpy as np
import numpy.typing as npt

import pref.utils
from pref.utils import (
    sigmoid,
    maximize,
    retrieve_reward_type, 
    retrieve_learning_algorithm,
    retrieve_social_welfare_function
)

from pref.pbo.base import ucb_function
from pref.pbo.confidence_sets import Ball

Action = npt.NDArray[float]
Utilities = Callable[[Action], float]
Reward = Callable[[Action], float]

RewardFunction = Callable[Utilities, Callable[Action, Reward]]
LearningAlgorithm = Callable[[RewardFunction, int], Tuple[Action, Reward]]

OmegaConf.register_new_resolver('div', lambda x, y: x/y)
OmegaConf.register_new_resolver(
    "generate_random_seed", pref.utils.generate_random_seed, use_cache=True
)

def pref_social_welfare_generator(config: Union[DictConfig, Dict]):
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Use common seed to allow for reproducibility
    random_seed = config.get("seed", pref.utils.generate_random_seed())
    pref.utils.seed_everything(random_seed)
    
    numpy_rng = pref.utils.Random.numpy_rng
    
    sw = retrieve_social_welfare_function(config.get("social_welfare_function", "beale"))
    social_welfare, actions_dim = sw.fun, sw.input_dim
    
    assert actions_dim is not None
    
    learn_utilities = config.get("learn_utilities", False)
    if learn_utilities:
        utilities_strategy = instantiate(config.utilities_strategy)
        utilities_strategy.init(actions_dim)
    
    input_dim = utilities_strategy.dimensions if learn_utilities else actions_dim
    
    initial_point = np.array([0] * input_dim)
    
    # define horizon (maximum number of turns)
    T = config.get("horizon", 30)

    confidence_set = Ball(
        kernel=config.get("kernel_type", "linear"),
        kernel_lengthscale=config.get("kernel_lengthscale", 2),
        bound=config.get("RKHS_bound", 0.2),
        beta=config.get("beta", 2),
        input_dim=input_dim
    )
    confidence_set.initial(initial_point)
    
    # quick fix for no initial data
    # TODO: Make sure to do proper initialization instead
    confidence_set = confidence_set.update(np.array([-3.4] * input_dim), 1)
    
    reward_type: RewardFunction = retrieve_reward_type(config.get("reward_type", "tw"))
    learning_algorithm: LearningAlgorithm = retrieve_learning_algorithm(config.get("learning_algorithm", "hedge"))

    last_actions = np.array([0] * actions_dim) # TODO: change this
    for t in range(T):
        # find next point to query
        
        s_ucb = [
            ucb_function(confidence_set, i=i, marginal=config.get("marginal_ucb", False))
            for i in range(input_dim)
        ]
        
        # define rewards from ucb
        
        if learn_utilities:
            discrete_utilities = maximize(s_ucb, utilities_strategy.dimensions, np.arange(0, 2, 1))
            utilities = utilities_strategy.convert(discrete_utilities[0])
        else:
            utilities = s_ucb

        reward_functions = reward_type(utilities)
        
        # run the rational agents
        
        actions, ucb_rewards = learning_algorithm(reward_functions, actions_dim)
        
        # measure preference
        
        probability = sigmoid(social_welfare(actions) - social_welfare(last_actions))
        preference = numpy_rng.binomial(1, probability)
        
        confidence_set = confidence_set.update(discrete_utilities[0] if learn_utilities else actions, preference)
        
        last_actions = actions
        rewards = social_welfare(actions)
        
        yield (s_ucb, actions, ucb_rewards, rewards, preference)

def pref_social_welfare(config: Union[DictConfig, Dict], generator=False):
    generator_object = pref_social_welfare_generator(config)
    
    if generator:
        return generator_object
    
    for _ in generator_object:
        pass