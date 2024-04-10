import numpy as np

from pref.utils import define_learning_algorithm, Random

@define_learning_algorithm(["hedge", "mw", "multiplicative_weights"])
def hedge(function, input_dim, max_iters=30, epsilon=0.1):
    # TODO: adapt the hard-coded values to the problem being solved
    
    # we first discretize the actions
    min_value = -8
    max_value = 8
    k = 20 # number of discrete actions to consider
    
    delta = (max_value - min_value) / (k-1)
    
    Q = np.ones((input_dim, k))
    
    for t in range(max_iters + 1):
        exp_Q = np.exp(Q)
        policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        actions_index = np.apply_along_axis(
            func1d=lambda p: Random.numpy_rng.choice(a=k, p=p),
            axis=1,
            arr=policy
        )

        actions = min_value + delta * actions_index
        
        # TODO: make sure this maps makes sense for the algorithm
        # it tries to map an arbitrary real value to something useful for the weights update
        output = np.exp(-np.exp(function(actions) / 2))
        
        if t == max_iters:
            return actions, output
        
        Q[np.arange(input_dim), actions_index] *= np.exp(-epsilon * output)