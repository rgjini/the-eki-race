'''
Modified Lorenz 96 model with spatially dependent forcing
By: Rebecca Gjini 

This modified version of the L96 model is meant to be used for testing EKI algorithms
-For parameter estimation of 40 parameters
'''
import numpy as np
import numba 
from numba import jit, njit

@njit
def lorenz(gamma, x): 
    '''
    This method computes the lorenz 96 system and ouputs the ode vector x
    
    gamma : array of floats
        forcing 
    x : array of floats
        Vector of the conditions to simulate through the lorenz system (i.e initial coniditions)

    Returns
    --------

    1D array of floats : 
        Vector of the solution to the L96 equation
    '''
    N = len(x)
    out_x = np.zeros((N))
    for kk in range(0, N): 
        out_x[kk] = - x[kk] + (x[(kk + 1)%N] - x[(kk - 2)])*x[(kk - 1)] + gamma[kk]
    return out_x

@njit
def runge_kutta_v(gamma, x_0, t, T):
    '''
    This method simulates the lorenz system over time using a 4th order runge-kutta scheme
    
    gamma : array of floats
        forcing 
    x_0 : array of floats
        Vector of the initial conitions to simulate through the lorenz system (i.e initial coniditions)
    t : double 
        time step
    T : int
        total time to simulate through

    Returns
    -------

    solv : 2d array of floats 
        rows are values at each time step and columns are the parameter vector of the 
        lorenz 96 model
    '''
    solv = np.empty((int(T / t) + 1, len(x_0))) #array to store the solution
    solv[0] = x_0
    for i in range(0, len(solv) - 1, 1): 
        #calculating k values for xn + 1
        k1 = lorenz(gamma, solv[i])
        k2 = lorenz(gamma, solv[i] + t*0.5*k1)
        k3 = lorenz(gamma, solv[i] + t*0.5*k2)
        k4 = lorenz(gamma, solv[i] + t*k3)
        #calculating the x_{n+1} value 
        solv[i + 1] = solv[i] + (1.0/6.0)*t*(k1 + 2*k2 + 2*k3 + k4)
    return solv

def G(gamma, x_0, t, T, ic_cov_sqrtt): 
    '''
    This function is the forward model for the mL96 problem, outputting the statistics of the state
    
    gamma : array of floats 
        forcing 
    x_0 : array of floats 
        initial condition (i.e. parameters) of the l96 system 
    t : double 
        time step
    T : int
        total time to simulate through
    ic_cov_sqrtt : 2D array of floats
        Square root of the covaraince to add perturb the initial condition

    Returns
    -------

    array of floats 
        time-averaged statistics of the states
    '''
    nx = len(x_0)
    y = np.zeros(2*nx)
    out = runge_kutta_v(gamma, x_0 + ic_cov_sqrtt@np.random.normal(0, 1, size = nx), t, T)[int(4/t):]
    y[:nx] = np.mean(out, axis = 0)
    y[nx:] = np.sqrt(np.var(out, axis = 0, ddof = 1))
    return y

def r(gamma, x_0, t, T, ic_cov_sqrtt, y, Rinv_sqrt, mu, Bsqrt): 
    '''
    r(x) function (vector form of cost function) for the levenburg-marquard algorithm with 
    finite differencing
    
    gamma : array of floats 
        forcing 
    x_0 : array of floats 
        initial condition (i.e. parameters) of the l96 system
    t : double 
        time step
    T : int
        total time to simulate through
    y : array of floats 
        the data
    Rin_sqrt : 2D array of floats
        Inverse squareroot of the data covaraince matrix
    mu : array of floats 
        prior mean
    Bsqrt : 2D array of floats
        Squareroot of the prior covaraince matrix

    Returns
    -------

    array of floats 
        Cost function vector
    
    '''
    return (1/np.sqrt(2))*np.concatenate((Rinv_sqrt@(G(Bsqrt@gamma + mu, x_0, t, T, ic_cov_sqrtt) - y), gamma))


