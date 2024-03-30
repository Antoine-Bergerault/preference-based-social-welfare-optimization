import numpy as np
from tqdm import tqdm

from pref import pref_social_welfare
from pref.games.base import stack_exchange

from visualize.base import surface, add_marker
import plotly.graph_objects as go

x, y = np.mgrid[-12:12:0.8, -12:12:0.8]

fig = surface(x, y, stack_exchange)
fig.layout.title = 'Welfare function'

# show surface of welfare function 
# fig.show()

T = 20
gen = pref_social_welfare({
    "social_welfare_function": "stack_exchange", 
    "learning_algorithm": "oracle",
    "RKHS_bound": 10, # control magnitudes
    "beta": 8e-3,
    "kernel_type": "gaussian",
    "kernel_lengthscale": 5, # control smoothness
    "horizon": T
}, generator=True)

rewards_hist = []
k = 0
for (s_ucb, actions, ucb_rewards, rewards, preference) in tqdm(gen, total=T):
    k+=1
    
    print("actions", actions)
    print("ucb_rewards", ucb_rewards)
    print("rewards", rewards)
    print("preference", preference)
    
    rewards_hist.append(rewards)
    
    add_marker([*actions, stack_exchange(actions)], fig)
    
    if k > 3:
        breakpoint()

    # show welfare function with markers:
    # fig.show()
    
    # show current UCB estimate of the social welfare function:
    # surface(x, y, s_ucb, parallelize=False).show()

t = np.arange(T)

fig = go.Figure(go.Scatter(x=t, y=np.array(rewards_hist), line_shape='spline'))
fig.show()