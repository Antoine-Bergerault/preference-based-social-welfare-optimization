from numpy import exp, sin, sqrt, pi
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

@define_social_welfare_function("eggholder", input_dim=2)
def eggholder(x):
    x_1, x_2 = x[0], x[1]
    
    # change range of function, usually evaluated in a square 512x512
    ##x_1 = x_1 * 512/12
    ##x_2 = x_2 * 512/12
    
    return -(x_2 + 47) * sin(sqrt(abs(x_2 + x_1/2 + 47))) - x_1 * sin(sqrt(abs(x_1 - (x_2 + 47))))

@define_social_welfare_function("cross_in_tray", input_dim=2)
def cross_in_tray(x):
    x_1, x_2 = x[0], x[1]
    
    return -0.0001 * (abs(sin(x_1) * sin(x_2) * exp(abs(100 - sqrt(x_1**2 + x_2**2)/pi))) + 1)**0.1

@define_social_welfare_function("bukin", input_dim=2)
def bukin(x):
    x_1, x_2 = x[0], x[1]
    
    return 100 * sqrt(abs(x_2 - 0.01 * x_1**2)) + 0.01 * abs(x_1 + 10)

class Game:
    pass