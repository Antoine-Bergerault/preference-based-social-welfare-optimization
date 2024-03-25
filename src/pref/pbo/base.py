from pref.pbo.confidence_sets import Ball

from math import sqrt, pi
import numpy as np

from casadi import Opti, vertcat, DM, dot, log, exp

def loss(z, pref):
    z_1t = z[1:]
    z_0 = z[:-1]
    
    ones = np.ones((z_1t.shape[0], z_1t.shape[1]))
    return dot(z_1t, pref) + dot(z_0, 1 - pref) - dot(ones, log(exp(z_1t) + exp(z_0)))

def compute_MLE_loss(t, pref_hist, K_0t, B, lambda_):
    opti = Opti()

    z_hist = opti.variable(t, 1)

    objective = loss(z_hist, pref_hist)

    opti.minimize(-objective) # maximize objective
    
    matrix_inv = opti.parameter(t, t)
    opti.set_value(matrix_inv, DM(np.linalg.inv(K_0t + lambda_ * np.identity(t))))
    
    opti.subject_to(z_hist.T @ matrix_inv @ z_hist <= B**2)

    # Silent ipopt solver
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', opts)
    
    sol = opti.solve()

    return sol.value(objective)

def beta1(epsilon, delta, t, B, confidence_set):
    # equation (7)
    C = 1 + 2/(1 + exp(-2*B))
    return sqrt(32 * t * (B**2) * log(
        (pi**2 * t**2 * confidence_set.covering_number(epsilon)) / (6 * delta)
    )) + C * epsilon * t

def ucb_function(confidence_set: Ball):
    x_hist = confidence_set.hist_data
    
    pref_hist = confidence_set.pref_data
    t = len(x_hist)
    
    B = confidence_set.bound
    lambda_ = 0.2 # regularization factor
    
    k = confidence_set.kernel
    K_0t = k(x_hist, x_hist)
    
    loss_MLE = compute_MLE_loss(t, pref_hist, K_0t, B, lambda_)
    beta1_t = beta1(0.2, 0.1, t, B, confidence_set)
    def f(x):
        # Implementing equation (21)
        K_x = k(x_hist, np.array([x]))
        k_xx = k(np.array([x]), np.array([x]))
        
        K_0x = np.block([
            [K_0t,     K_x],
            [K_x.T,     k_xx]
        ])

        opti = Opti()

        z_hist = opti.variable(t)
        z = opti.variable()
        
        z_plus = vertcat(z_hist, z)

        objective = z - z_hist[-1]

        opti.minimize(-objective) # maximize objective
        
        matrix_inv = opti.parameter(t+1, t+1)
        opti.set_value(matrix_inv, DM(np.linalg.inv(K_0x + lambda_ * np.identity(t+1))))
        
        opti.subject_to(z_plus.T @ matrix_inv @ z_plus <= B**2)
        
        # in the paper: loss(z_hist, pref_hist) >= loss_MLE - beta1_t
        # TODO: I don't know why this equation makes sense, any loss should be greater than loss MLE
        opti.subject_to(loss(z_hist, pref_hist) <= loss_MLE + beta1_t)

        # Silent ipopt solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)
        
        sol = opti.solve()

        return sol.value(objective)
    
    return f
