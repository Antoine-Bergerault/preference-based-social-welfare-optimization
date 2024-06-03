from pref.utils import retrieve_social_welfare_function
from visualize.base import surface

import numpy as np
x, y = np.mgrid[-12:12:0.8, -12:12:0.8]

for social_welfare_function in ["eggholder"]:
    print(social_welfare_function)
    
    fn = retrieve_social_welfare_function(social_welfare_function).fun

    fig = surface(x, y, fn)
    fig.layout.title = social_welfare_function

    # show surface of welfare function 
    fig.show()
    
    breakpoint()