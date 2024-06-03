import numpy as np

from pref.utils import define_learning_algorithm, Random

@define_learning_algorithm(["hedge", "mw", "multiplicative_weights"])
def hedge(utilities, input_dim, max_iters=10, epsilon=0.1):
    # TODO: adapt the hard-coded values to the problem being solved
    if not isinstance(utilities, list):
        utilities = [utilities] * input_dim
    
    # we first discretize the actions
    min_value = -8
    max_value = 8
    k = 20 # number of discrete actions to consider
    
    delta = (max_value - min_value) / (k-1)
    
    Q = np.ones((input_dim, k))
    
    # TODO: make sure this maps makes sense for the algorithm
    # it tries to map an arbitrary real value to something useful for the weights update
    def loss_map(x):
        return np.exp(-np.exp(x / 2))
    
    for t in range(max_iters + 1):
        exp_Q = np.exp(Q)
        policy = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        
        chosen_actions_indices = np.apply_along_axis(
            func1d=lambda p: Random.numpy_rng.choice(a=k, p=p),
            axis=1,
            arr=policy
        )
        
        # return after max_iters updates
        if t == max_iters:
            chosen_actions = min_value + delta * chosen_actions_indices
            outputs = [loss_map(utility(chosen_actions)) for utility in utilities]
            return chosen_actions, outputs
        
        # repeat players and actions indices to cover all possibilities
        players_indices = np.repeat(np.arange(input_dim), k)
        actions_indices = np.tile(np.arange(k), input_dim)
        
        # all unilateral changes including chosen actions (i.e. trivial changes)
        unilateral_changes = np.kron(chosen_actions_indices, np.ones((input_dim, 1))) * (1 - np.eye(input_dim)) # masked actions indices
        unilateral_changes = np.kron(unilateral_changes, np.ones((k, 1))) + np.kron(np.eye(input_dim), np.arange(k)).T # add unilateral changes
        unilateral_changes = min_value + delta * unilateral_changes
        
        # update the Q values according to the correct utilities
        for i in range(unilateral_changes.shape[0]):
            player = players_indices[i]
            player_actions = actions_indices[i]
            
            output = loss_map(utilities[player](unilateral_changes[i]))
            
            Q[player, player_actions] *= np.exp(-epsilon * output)