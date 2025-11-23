'''
Modified Lorenz 96 model with approximated 
forcing using a neural network
By: Rebecca Gjini 

This modified version of the L96 model is meant to be used for testing EKI algorithms
-For parameter estimation of neural net weights and biases
'''
import numpy as np
import torch
import torch.nn as nn
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
def runge_kutta_v(gamma, x_0, t, T, st = 2):
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
    st : int
        number of samples to take from time series (default, every other element of time series)

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
    return solv[::st]


def G(weights, net, net_input, x_0, t, T, ic_cov_sqrtt, st = 2): 
    '''
    This function is the forward model for the mL96 problem, outputting the statistics of the state
    
    weights : array of floats 
        neural network weights and biases 
    net : pytorch neural network object
        neural network object with 1 hidden layer with 10 neurons 
    net_input : pytorch tensor
        values to put into neural network to make forcing prediction
    x_0 : array of floats 
        initial condition (i.e. parameters) of the l96 system 
    t : double 
        time step
    T : int
        total time to simulate through
    ic_cov_sqrtt : 2D array of floats
        Square root of the covaraince to add perturb the initial condition
    st : int
        number of samples to take from time series (default, every other element of time series)

    Returns
    -------

    array of floats 
        time-averaged statistics of the states
    '''
    nx = len(x_0)
    y = np.zeros(2*nx)
    load_weights(net, weights)
    gamma = (8 + 6*net(net_input).detach().numpy())[:,0]
    out = runge_kutta_v(gamma, x_0 + ic_cov_sqrtt@np.random.normal(0, 1, size = nx), t, T)[int(4/(st*t)):]
    y[:nx] = np.mean(out, axis = 0)
    y[nx:] = np.sqrt(np.var(out, axis = 0, ddof = 1))
    return y

def r(weights, net, net_input, x_0, t, T, ic_cov_sqrtt, y, Rinv_sqrt, mu, Bsqrt, st = 2): 
    '''
    r(x) function (vector form of cost function) for the levenburg-marquard algorithm with 
    finite differencing
    
    weights : array of floats 
        neural network weights and biases 
    net : pytorch neural network object
        neural network object with 1 hidden layer with 10 neurons 
    net_input : pytorch tensor
        values to put into neural network to make forcing prediction
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
    st : int
        number of samples to take from time series (default, every other element of time series)

    Returns
    -------

    array of floats 
        Cost function vector
    
    '''
    return (1/np.sqrt(2))*np.concatenate((Rinv_sqrt@(G(Bsqrt@weights + mu, net, net_input, x_0, t, T, ic_cov_sqrtt, st) - y), weights))

#### Neural Network functions and objects 
#Define neural network object
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def load_weights(model, flattened_params):
    start = 0  # Keep track of the position in the flattened array

    with torch.no_grad():  # Disable gradient tracking during manual weight assignment
        for param in model.parameters():
            param_length = param.numel()  # Number of elements in the parameter tensor
            new_values = flattened_params[start:start + param_length].reshape(param.shape)
            param.copy_(torch.tensor(new_values, dtype=torch.float32))
            start += param_length  # Move the index forward
