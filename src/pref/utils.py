import functools
from types import SimpleNamespace

from numpy import exp

## Common functions

def sigmoid(x):
    # stable implementation of the sigmoid function
    exp_ = exp(x)
    
    if exp_ == float("inf"):
        return 1
    
    return exp(x)/(1 + exp(x))

## Registering and retrieving objects using configuration keywords

def registry_decorator(registry, **additional_parameters):
    def decorator_factory(names, **kwargs):
        # only preserve pre-declared keys
        additional_parameters.update((k, v) for (k, v) in kwargs.items() if k in additional_parameters)
        
        if not isinstance(names, list):
            names = [names]

        def decorator(function):
            if len(additional_parameters) > 0:
                data = SimpleNamespace(fun=function, **additional_parameters)
            else:
                data = function
            
            for name in names:
                registry[name] = data
            
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)
            
            return wrapper

        return decorator
    
    return decorator_factory


reward_types_registry = {}
define_reward_type = registry_decorator(reward_types_registry)

def retrieve_reward_type(reward_type: str):
    # make sure to register first any reward type
    import pref.learning
    
    if reward_type in reward_types_registry:
        return reward_types_registry[reward_type]
    
    raise ValueError(f'''
        Reward type function "{reward_type}" not found. 
        Make sure to declare it with the @define_reward_type decorator and import it.
        
        Available types: {', '.join(list(reward_types_registry.keys()))}.
    ''')


learning_algorithms_registry = {}
define_learning_algorithm = registry_decorator(learning_algorithms_registry)

def retrieve_learning_algorithm(learning_algorithm: str):
    # make sure to register first any learning algorithm
    import pref.learning
    
    if learning_algorithm in learning_algorithms_registry:
        return learning_algorithms_registry[learning_algorithm]
    
    raise ValueError(f'''
        Learning algorithm "{learning_algorithm}" not found. 
        Make sure to declare it with the @define_learning_algorithm decorator and import it.
        
        Available algorithms: {', '.join(list(learning_algorithms_registry.keys()))}.
    ''')


social_welfare_functions_registry = {}
define_social_welfare_function = registry_decorator(social_welfare_functions_registry, input_dim=None)

def retrieve_social_welfare_function(social_welfare_function: str):
    # make sure to register first any social welfare function
    import pref.games
    
    if social_welfare_function in social_welfare_functions_registry:
        return social_welfare_functions_registry[social_welfare_function]
    
    raise ValueError(f'''
        Social welfare function "{social_welfare_function}" not found. 
        Make sure to declare it with the @define_social_welfare_function decorator and import it.
        
        Available functions: {', '.join(list(social_welfare_functions_registry.keys()))}.
    ''')