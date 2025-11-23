'''
Lorenz 63 model 
By: Rebecca Gjini 

This version of the L63 model is meant to be used for testing EKI algorithms
-For parameter estimation of 2/3 parameters (rho and beta)
'''

#Import statements
import numpy as np
import numba
from numba import jit, njit
from scipy.linalg import sqrtm

@njit
def lorenz(theta, x): 
    '''
    This method computes the lorenz 63 system and ouputs the ode vector [x, y, z]
    
    theta : array of floats
        Vector of the three parameters sigma, rho, beta
    x : array of floats
        Vector of the conitions to simulate through the lorenz system (i.e initial coniditions)

    Returns
    --------

    1D array of floats : 
        Vector of the solution to the L63 equations 
    '''
    
    return np.array([(theta[0]*(x[1] - x[0])), 
                    (x[0]*(theta[1] - x[2]) - x[1]), 
                    (x[0]*x[1] - theta[2]*x[2])])

@njit
def runge_kutta_v(theta, x_0, t, total_T): 
    '''
    This method simulates the lorenz system over time using a 4th order runge-kutta scheme
    
    theta : array of floats
        Vector of the three parameters sigma, rho, beta
    x_0 : array of floats
        Vector of the initial conitions to simulate through the lorenz system (i.e initial coniditions)
    t : double 
        time step
    total_T : int
        total time to simulate through

    Returns
    -------

    solv : 2d array of floats 
        rows are values at each time step and columns are the 3 parameter of the 
        lorenz 63 model
    '''
    solv = np.zeros((int(total_T/t) + 1,3))  #solution array

    solv[0] = x_0                        #insert initial conditions here
    for i in range(0, len(solv) - 1, 1):
        #calculating k values for x + 1
        k1 = lorenz(theta, solv[i])
        k2 = lorenz(theta, solv[i] + t*0.5*k1)
        k3 = lorenz(theta, solv[i] + t*0.5*k2)
        k4 = lorenz(theta, solv[i] + t*k3)
        solv[i + 1] = solv[i] + (1.0/6.0)*t*(k1 + 2*k2 + 2*k3 + k4)
    return solv

def G(rb, sigma, x_0, ic_cov_sqrtt, t, total_T): 
    '''
    This function create the statistics vector from the Lorenz 63 simulation
    
    rb : tuple of doubles 
        rho and beta parameters for the lorenz 63 system
    sigma : double
        sigma parameter for the lorenz 63 system
    x_0 : array of floats
        Vector of the initial conitions to simulate through the lorenz system (i.e initial coniditions)
    ic_cov_sqrtt : 2D array of floats
        Square root of the covaraince to add perturb the initial condition
    t : double 
        time step
    total_T : int
        total time to simulate through

    Returns
    -------

    gg : array of floats 
        Statistical information about the lorenz 63 attractor
    '''
    rho = np.exp(rb[0])
    beta = np.exp(rb[1])
    theta = np.array([sigma, rho, beta])  #initialize parameter vector
    RK = runge_kutta_v(theta, x_0 + ic_cov_sqrtt@np.random.normal(0, 1, size = 3),
                       t, total_T)[int(30/t):]   #run through runge-kutta timestepping 
    gg = np.zeros(9)
    gg[0:3] = np.mean(RK, axis = 0)  #calculate the mean for the first three points
    RK_cov = np.cov(RK.T)
    gg[3:6] = np.diag(RK_cov)     # calculate the varinaces for the next three points
    gg[6:8] = RK_cov[0,1:]        #calculate the covarinaces for the last three points
    gg[8] =  RK_cov[1,2]
    return gg


def r(rb, sigma, x_0, ic_cov_sqrtt, t, total_T, y, Rinv_sqrt, mu, Bsqrt):   #Binv_sqrt
    '''
    r(x) function (vector form of cost function) for the levenburg-marquard with 
    finite differencing
    
    rb : tuple of doubles 
        rho and beta parameters for the lorenz 63 system
    sigma : double
        sigma parameter for the lorenz 63 system
    x_0 : array of floats
        Vector of the initial conitions to simulate through the lorenz system (i.e initial coniditions)
    ic_cov_sqrtt : 2D array of floats
        Square root of the covaraince to add perturb the initial condition
    t : double 
        time step
    total_T : int
        total time to simulate through
    y : array of floats 
        the data
    Rin_sqrt : 2D array of floats
        Inverse squareroot of the data covaraince matrix
    mu : array of floats 
        prior mean
    Bsqrt : 2D array of floats
        Squareroot of the prior covaraince matrix
    Bin_sqrt : 2D array of floats
        Inverse squareroot of the prior covaraince matrix

    Returns
    -------

    array of floats 
        Cost function vector
    '''
    return (1/np.sqrt(2))*np.concatenate((Rinv_sqrt@(G(Bsqrt@rb + mu, sigma, x_0, ic_cov_sqrtt, t, total_T) - y), rb))

