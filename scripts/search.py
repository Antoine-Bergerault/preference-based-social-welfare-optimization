from pref import pref_social_welfare

import random
import string
import wandb
from tqdm import tqdm
from functools import reduce

hyperparameters = {
    "RKHS_bound": [6, 8, 10],
    "kernel_lengthscale": [2.5, 4, 5],
    "beta": [8e-3, 5e-2, 5e-1]
}

def get_params(possibilities, args={}):
    if len(possibilities) == 0:
        yield args
    else:
        item = possibilities.popitem()
        for value in item[1]:
            args[item[0]] = value
            yield from get_params(possibilities, args)
        possibilities[item[0]] = item[1]


baseconfig = {
    "social_welfare_function": "stack_exchange", 
    "learning_algorithm": "oracle",
    "kernel_type": "gaussian",
    "horizon": 15
}

unique_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

number_of_experiments = reduce(lambda a,b: a*b, [len(v) for v in hyperparameters.values()])
for params in tqdm(get_params(hyperparameters), total=number_of_experiments):
    config = baseconfig | params

    assert "social_welfare_function" in config
    
    gen = pref_social_welfare(config, generator=True)

    run = wandb.init(
        project=("sycamore-project-search-" + config["social_welfare_function"]),
        group=unique_name,
        tags=[
            ("kernel_" + config["kernel_type"])
        ],
        
        config=config
    )
    
    try:
        for (s_ucb, actions, ucb_rewards, rewards, preference) in gen:                        
            run.log({
                "ucb_rewards": ucb_rewards,
                "rewards": rewards
            }, commit=True)
            
        run.finish()
    except Exception as e:
        print(e)
        run.finish(exit_code=1)