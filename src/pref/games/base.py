from numpy import exp
from pref.utils import define_social_welfare_function

@define_social_welfare_function("beale", input_dim=2)
def beale(x):
    x_1, x_2 = x[0], x[1]
    
    # expand function (note: don't modify in-place)
    x_1 = x_1 / 5
    x_2 = x_2 / 5
    
    value = (1.5 - x_1 + x_1*x_2)**2 + (2.25 - x_1 +x_1*x_2**2)**2 + (2.625 - x_1 + x_1*x_2**3)**2
    
    return -value

@define_social_welfare_function("stack_exchange", input_dim=2)
def stack_exchange(x):
    x_1, x_2 = x[0], x[1]
    
    # expand function (note: don't modify in-place)
    x_1 = x_1 / 5
    x_2 = x_2 / 5
    
    return 3 * (1 - x_1)**2 * exp(-x_1**2 - (x_2+1)**2) - 10*(x_1/5 - x_1**3 - x_2**5) * exp(-x_1**2 - x_2**2) - 1/3 * exp(-(x_1+1)**2 - x_2**2)

class Game:
    pass