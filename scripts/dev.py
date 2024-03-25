import numpy as np

from pref import pref_social_welfare
from pref.games.base import stack_exchange

from visualize.base import surface, add_marker

x, y = np.mgrid[-4:4:0.1, -4:4:0.1]

fig = surface(x, y, stack_exchange)

fig.update_layout(
        title='Welfare function', autosize=False,
        width=800, height=800,
        margin=dict(l=80, r=50, b=65, t=90)
)

fig.show()

k = 0
gen = pref_social_welfare({
    "social_welfare_function": "stack_exchange", 
    "learning_algorithm": "oracle"
}, generator=True)

for (s_ucb, actions, rewards, preference) in gen:
    print("actions", actions)
    print("rewards", rewards)
    print("preference", preference)
    
    add_marker([*actions, stack_exchange(actions)], fig)
    fig.show()
    
    # show current UCB estimate of the social welfare function
    surface(x, y, s_ucb, parallelize=False).show()
    
    k+=1
    
    breakpoint()