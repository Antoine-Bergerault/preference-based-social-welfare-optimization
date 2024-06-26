from pref.pbo.confidence_sets import Ball

from math import sqrt
import numpy as np

from casadi import Opti, vertcat, DM, dot, log, exp

def log_likelihood(z, pref):
    z_1t = z[1:]
    z_0 = z[:-1]
    
    ones = np.ones((z_1t.shape[0], z_1t.shape[1]))
    return dot(z_1t, pref) + dot(z_0, 1 - pref) - dot(ones, log(exp(z_1t) + exp(z_0)))

def compute_MLE_log_likelihood(t, pref_hist, K_0t, B, lambda_):
    opti = Opti()

    z_hist = opti.variable(t, 1)

    objective = log_likelihood(z_hist, pref_hist)

    opti.minimize(-objective) # maximize objective
    
    matrix_inv = opti.parameter(t, t)
    opti.set_value(matrix_inv, DM(np.linalg.inv(K_0t + lambda_ * np.identity(t))))
    
    opti.subject_to(z_hist.T @ matrix_inv @ z_hist <= B**2)

    # Silent ipopt solver
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', opts)
    
    sol = opti.solve()

    return sol.value(objective)

def beta1(t, confidence_set):
    beta = confidence_set.beta
    return beta * sqrt(t+1)

def ucb_function(confidence_set: Ball, i, marginal=False):
    x_hist = confidence_set.hist_data
    
    pref_hist = confidence_set.pref_data
    t = len(x_hist)
    
    B = confidence_set.bound
    lambda_ = 0.2 # regularization factor
    
    k = confidence_set.kernel
    K_0t = k(x_hist, x_hist)
    
    log_likelihood_MLE = compute_MLE_log_likelihood(t, pref_hist, K_0t, B, lambda_)
    beta1_t = beta1(t, confidence_set)
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
        
        matrix_inv = opti.parameter(t+1, t+1)
        opti.set_value(matrix_inv, DM(np.linalg.inv(K_0x + lambda_ * np.identity(t+1))))
        
        opti.subject_to(z_plus.T @ matrix_inv @ z_plus <= B**2)
        
        opti.subject_to(log_likelihood(z_hist, pref_hist) >= log_likelihood_MLE - beta1_t)

        if marginal:
            # maximizing the optimistic marginal contribution
            masked_x = np.array(x)
            masked_x[i] = 0
            D = matrix_inv @ k(np.vstack([x_hist, np.array(x)]), masked_x[None, ...])
            objective = z - z_plus.T @ D
        else:
            # corresponding to classical preferential bayesian optimization
            objective = z - z_hist[-1]

        opti.minimize(-objective) # maximize objective
        
        # Silent ipopt solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)
        
        sol = opti.solve()

        return sol.value(objective)
    
    return f
