import numpy as np
from numpy import pi
from numpy.random import normal

def derivs(t, y, params): # here is all the physics: the left side of the equation

    eps, a_s = params['eps'], params['a_s']
    K, a_c = params['K'], params['a_c']

    F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']
    Famp, alpha = eps*pi/a_s, 2*pi/a_s # Shortcuts for derivatives
    gamma, brand = params['gamma'], params['brand']

    neq=len(y)
    nhalf=int(neq/2)
    deriv=[0.]*neq  # just to initialize the new array

    for i in range(nhalf):  # the second half of y is velocities
        deriv[i]=y[i+nhalf] # this enables the mapping of Newton to 1st order

    for i in range(1,nhalf-1):
        deriv[i+nhalf]= F_ext - Famp*np.sin(alpha*y[i]) + K*(y[i+1]+y[i-1]-2*y[i]) - gamma*deriv[i]

    # OBC
    # First particle
    i=0
    deriv[i+nhalf]= F_lhs + F_ext - Famp*np.sin(alpha*y[i]) + K*(y[i+1]-y[i]-a_c) - gamma*deriv[i]
    # Last particle
    i=nhalf-1
#    deriv[i+nhalf]= F_rhs + F_ext - Famp*np.sin(alpha*y[i]) - K*(y[i]-y[i-1]-a_c) - gamma*deriv[i]
    deriv[i+nhalf]= F_rhs + F_ext - Famp*np.sin(alpha*y[i]) + K*(y[i-1]-y[i]+a_c) - gamma*deriv[i]

    return deriv

def sub_en(y, params): # here is all the physics: the left side of the equation
    eps, a_s = params['eps'], params['a_s']
    return eps/2*(-1-np.cos(2*pi/a_s*y))

def spring_en(y, params): # here is all the physics: the left side of the equation
    K, a_c = params['K'], params['a_c']
    n = len(y)
    en = [K/2*(y[i+1]-y[i]-a_c)**2 for i in range(0,n-1)]
    return np.array(en)
