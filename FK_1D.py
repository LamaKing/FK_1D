import numpy as np
from numpy import pi
from numpy.random import normal

global t_eps
t_eps = 1 # Set to 0: turn off the substrate potential for debug purposes

def derivs(t, y, params): # here is all the physics: the left side of the equation

    # This assignment is a bit stupid and time-wasting, these should be global variables.
    #g, a_c = params['g'], params['a_c'] # Coupling and chain spacing
    a_c = params['a_c'] # chain spacing
    F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']
    gammav, brandv = params['gammav'], params['brandv'] # Gamma and brand t are vector, for thermal gradient
    gv = params['gv'] # atom-wise coupling, i.e. atom-wise spring.
    BC = params['BC'] # 1 for PBC 0 for OBC

    # Initialise arrays
    neq = len(y)
    neq2 = int(neq/2) # = Number of particles
    deriv = np.zeros(neq)  # the accelleration array
    noise = normal(0, 1, size=neq2) # Gaussian numbers

    #### POSITIONS DOT #####
    for i in range(neq2):  # the second half of y is velocities
        deriv[i] = y[i+neq2] # this enables the mapping of Newton to 1st order
    #print(deriv)

    #### VELOCITIES DOT ####
    #------- Chain bulk
    for i in range(1, neq2-1):
        deriv[i+neq2] = F_ext - t_eps*np.sin(y[i]) - gammav[i]*y[i+neq2] + brandv[i]*noise[i] \
            + gv[i]*(y[i+1] + y[i-1] - 2*y[i])
        #print(deriv)
    #------- Bboundary conditions
    PBC_term = y[neq2-1] - y[0] - (neq2-1)*a_c
    #------------ First particle
    i=0
    deriv[i+neq2] = F_ext - t_eps*np.sin(y[i]) - gammav[i]*y[i+neq2] + brandv[i]*noise[i] \
        + gv[i]*(y[i+1] - y[i] - a_c + BC*PBC_term) + F_lhs
    #print(deriv)
    #------------ Last particle
    i=neq2-1
    deriv[i+neq2] = F_ext - t_eps*np.sin(y[i]) - gammav[i]*y[i+neq2] + brandv[i]*noise[i] \
        - gv[i]*(y[i] - y[i-1] - a_c + BC*PBC_term) + F_rhs
    #print(deriv)
    #print('='*30)
    return deriv

def sub_en(y):
    return t_eps*(1-np.cos(y))

def spring_en(y, g, a_c, BC, params):
    gv = params['gv']
    Np = len(y)
    en = [gv[i]/2*(y[i+1]-y[i]-a_c)**2 for i in range(0,Np-1)]
    # If in PBC add last connection, otherwise 0 (keep array Np-long)
    en += [BC*gv[Np-1]/2*(y[0]+(Np-1)*a_c-y[Np-1])**2]
    return np.array(en)
