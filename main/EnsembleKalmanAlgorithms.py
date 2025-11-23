'''
Ensemble Kalman Algorithms
By: Rebecca Gjini

This script stores all the EKI algorithms to compete against each other
'''

import numpy as np 
from numpy.linalg import cholesky, solve, norm, svd

def matrix_inv(A): 
    '''
    Take the inverse of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the inverse of 

    Returns 
    -------

    A_inv : list of floats 2D array
        inverse of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    return (u_a/l_a)@u_a.T

def matrix_inv_sqrt(A): 
    '''
    Take the inverse sqrt of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the inverse sqrt of 

    Returns 
    -------

    A_inv_sqrt : list of floats 2D array
        inverse square root of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    return (u_a/np.sqrt(l_a))@u_a.T  

def matrix_sqrt(A): 
    '''
    Take the sqrt of a matrix using SVD

    A: list of floats 2D array
        Matrix to take the sqrt of 

    Returns 
    -------

    A_sqrt : list of floats 2D array
        square root of matrix A
    '''
    if A.ndim == 1: 
        return np.sqrt(A)
    else: 
        [u_a, l_a, q_a] = svd(A)  
        return (u_a*np.sqrt(l_a))@u_a.T
    
def matrix_inv_LR(A, tol): 
    '''
    Make low rank matrix inverse approximation using SVD

    A: list of floats 2D array
        Matrix to take the inverse of 

    Returns 
    -------

    A_inv : list of floats 2D array
        inverse approximation of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    for ii in range(1, len(l_a) + 1): 
        sum = np.sum(l_a[:ii]**2)
        tolerance = (1 - tol)*np.sum(l_a**2)
        if sum >= tolerance: 
            break
    return (q_a[:ii, :].T/l_a[:ii])@u_a[:,:ii].T

def matrix_LR(A, tol): 
    '''
    Make low rank matrix approximation using SVD

    A: list of floats 2D array
        Matrix to take the inverse of 

    Returns 
    -------

    A : list of floats 2D array
        low rank approximation of matrix A
    '''
    [u_a, l_a, q_a] = svd(A)  
    for ii in range(1, len(l_a)): 
        sum = np.sum(l_a[:ii]**2)
        tolerance = (1 - tol)*np.sum(l_a**2)
        if sum >= tolerance: 
            break
    return u_a[:,:ii]*l_a[:ii]@q_a[:ii, :]

def matrix_pseudo_inv(A): 
    '''
    Make pseudo inverse approximation using SVD

    A: list of floats 2D array
        Matrix to take the inverse of 

    Returns 
    -------

    A_inv : list of floats 2D array
        pseudo inverse of matrix A
    '''
    [u_a, l_a, q_a] = svd(A, full_matrices = False)  
    return (q_a.T/l_a)@u_a.T
    

def TEKI(func, u_0, args, y, R, mu, B, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    TEKI algorithm 

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: list of floats array
        data vector with dimensions (m, 1)
    R: list of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: list of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n, n)
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between func_n and func_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit 
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 # exit criteria 

    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0
    #Compute needed matrices
    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)
    #Date vector
    z = np.concatenate((Rsq_inv@y, np.zeros(N_t)))
    z_len = len(z)
    #Identity matrix (new data error matrix)
    I = np.identity(z_len)
    #Ensemble mean and initial RMSE
    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((z_len, K))
        for j in range(0, K): 
            g[:y_len,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
            g[y_len:,j] = u[i,:,j] 
        g_bar = np.mean(g, axis = 1)
        #Calculating covariances 
        Cug = ((u[i] - u_bar[:,np.newaxis])@(g.T - g_bar))/(K - 1)   #Cxy
        Cgg = np.cov(g)                                    #Cyy
        #Update
        update = Cug@solve(Cgg + I, (np.random.normal(0, 1, size = (z_len, K)) - 
                                    g) + z[:,np.newaxis])
        u[i + 1] = u[i] + update
        #Calculate ensemble mean
        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            #print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if (mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0):  #Convergence Criteria
                exit = 2
                break
        else: #tol_f 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break

    return (Bsq@u[i + 1] + mu[:,np.newaxis]), (i+1)*K, exit


def UKI(func, args, y, R, mu, B, a = 1, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    UKI algorithm with regularization.

    func: function 
        forward model function that is compared to the data
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: List of floats array
        data vector with dimensions (m, 1)
    R: List of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: List of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n,n)
    a: float
        adjustment factor
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between f_n and f_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    Note: Additional ways to calculate g_bar, using version where g_bar = g(m) = g[0]
     # g_bar = np.mean(g, axis = 1)
     # weighted_g = np.concatenate((g[:,0,np.newaxis]*(1 - (1/(a**2))), g[:,1:]/(2*(a**2)*N_t)), axis = 1)
     # g_bar = np.sum(weighted_g, axis = 1)
    ''' 
    exit = 0 #exit criteria 
    N_t = len(mu) 
    K = 2*N_t + 1 
    y_len = len(y)
    #Compure needed matrices
    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)  #always the same
    B_ = np.identity(N_t)  #B hat (change of variable B)
    B_sq = np.identity(N_t)  #iteratively changes

    c_j = a*np.sqrt(N_t)
    m = np.zeros((max_iter + 1, N_t))   #initialize array to store mean of parameter vector for each iteration
    #Sigma point calculation
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    for ii in range(0,N_t + 1): 
        if ii == 0:
            u[0,:,ii] = m[0]
        else: 
            u[0,:,ii] = m[0] + c_j*B_sq[:,ii - 1]
            u[0,:,ii + N_t] = m[0] - c_j*B_sq[:,ii - 1]
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@m[0] + mu, *args)))/np.sqrt(y_len)
    #Data vector
    z = np.concatenate((Rsq_inv@y, np.zeros(N_t)))
    z_len = len(z)
    #Identity matrix for data errors
    I = np.identity(z_len)

    #Algorithm
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((z_len, K))
        for j in range(0, K): 
            g[:y_len,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
            g[y_len:,j] = u[i,:,j]

        #Calculating covariances 
        Cug = ((u[i,:,1:] - m[i,:,np.newaxis])@(g[:,1:].T - g[:,0]))/(2*(a**2)*N_t)  #Cxy
        Cgg = ((g[:,1:] - g[:,0,np.newaxis])@(g[:,1:].T - g[:,0]))/(2*(a**2)*N_t)   #Cyy
        #Update
        update = Cug@solve(Cgg + I, (z - g[:,0])) 
        m[i + 1] = m[i] + update
        #Update covariance of prior
        B_ = B_ - Cug@solve(Cgg + I, Cug.T)
        B_sq = cholesky(B_) 
        #calculate new sigma points
        for jj in range(0,N_t + 1): 
            if jj == 0:
                u[i + 1,:,jj] = m[i + 1]
            else: 
                u[i + 1,:,jj] = m[i + 1] + c_j*B_sq[:,jj - 1]
                u[i + 1,:,jj + N_t] = m[i + 1] - c_j*B_sq[:,jj - 1]

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@m[i + 1] + mu, *args)))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(m[i + 1] - m[i])/norm(m[i])
            if ((mean_diff < tol_x) and (norm(m[i]) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@m[i + 1] + mu, *args) ))/np.sqrt(y_len)
            # print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse: #Converence Criteria 
                exit = 1
                break
        elif method == "tol_x":  
            mean_diff = norm(m[i + 1] - m[i])/norm(m[i])
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if ((mean_diff < tol_x) and (norm(m[i]) > 1e-5) and (i > 0)): #Converence Criteria 
                exit = 2
                break
        else: 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@m[i + 1] + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Converence Criteria 
                exit = 3
                break

    return (Bsq@u[i + 1] + mu[:,np.newaxis]), (i+1)*K, exit 


def ETKI(func, u_0, args, y, R, mu, B, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    ETKI algorithm inspired by the ETKF with regularization.

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: List of floats array
        data vector with dimensions (m, 1)
    R: List of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: List of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n,n)
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between f_n and f_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 #exit criteria

    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0

    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)

    II = np.identity(K)

    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    z = np.concatenate((Rsq_inv@y, np.zeros(N_t)))
    z_len = len(z)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((z_len, K))
        for j in range(0, K): 
            g[:y_len,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
            g[y_len:,j] = u[i,:,j]
        g_bar = np.mean(g, axis = 1)
        #Calculating perturbation matrices
        Cu = (u[i] - u_bar[:,np.newaxis])/np.sqrt(K - 1)   #X
        Cg = (g - g_bar[:,np.newaxis])/np.sqrt(K - 1)      #Y
        #Compute analysis weights
        w_a = solve(II + Cg.T@Cg, Cg.T@(z - g_bar))
        #Update analysis mean
        u_a = u_bar + Cu@w_a
        #Eigen Decomposition
        [Ue, Le, Qe] = svd(Cg.T@Cg)
        TT = (Ue/np.sqrt(Le + np.ones(K)))@Ue.T
        #Update
        u[i + 1] = u_a[:, np.newaxis] + np.sqrt(K-1)*(Cu@TT)    #update 
        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            #print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse: #Convergence criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)): #Convergence criteria
                exit = 2
                break
        else: 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence criteria
                exit = 3
                break

    return Bsq@u[i + 1] + mu[:,np.newaxis], (i+1)*K, exit 

def IEKF(func, u_0, args, y, R, mu, B, alpha = 1, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    Iterative ensemble Kalman filter (IEKF) with statistical regularization algorithm from Chada et al. 2021 

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: List of floats array
        data vector with dimensions (m, 1)
    R: List of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: List of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n,n)
    alpha: float
        adjustment factor
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between f_n and f_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 # exit criteria 
    
    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0

    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)

    I = np.identity(y_len)

    #Prior mean \mu
    mu_t = np.zeros((N_t, K))  #needs to be able to be added to each ensemble member

    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((y_len, K))
        for j in range(0, K): 
            g[:,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
        g_bar = np.mean(g, axis = 1)

        #Calculating covariances
        Cug = ((u[i] - u_bar[:,np.newaxis])@(g.T - g_bar))/(K - 1)  #Cxy
        Cuu = ((u[i] - u_bar[:,np.newaxis])@(u[i].T - u_bar))/(K - 1) #Cxx np.cov(u[i])
        #Add noise to \mu
        mu_0 = mu_t + (np.sqrt(2/alpha))*np.random.normal(0, 1, size = (N_t, K))
        #Update for IEKF w/ SL
        A = ((I*np.sqrt(2/alpha))@np.random.normal(0, 1, size = (y_len, K)) - g) + Rsq_inv@y[:,np.newaxis] - Cug.T@solve(Cuu, mu_0 - u[i])
        Sigma = Cug.T@solve(Cuu, solve(Cuu, Cug))
        update = solve(Cuu, Cug@solve(Sigma + I, A))

        u[i + 1] = (1-alpha)*u[i] + alpha*(mu_0 + update)

        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            # print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args) ))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse: #Convergence criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)): #Convergence criteria
                exit = 2
                break
        else: 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence criteria
                exit = 3
                break

    return Bsq@u[i + 1] + mu[:,np.newaxis], (i+1)*K, exit 

def IEKF_stable(func, u_0, args, y, R, mu, B, alpha = 1, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    Iterative ensemble Kalman filter (IEKF) with statistical regularization from Chada et al. (with derivative 
    calculated using the Chen and Oliver 2013 ensemble gradient approximation for stability)

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: List of floats array
        data vector with dimensions (m, 1)
    R: List of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: List of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n,n)
    alpha: float
        adjustment factor
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between f_n and f_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 # exit criteria 
    
    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0

    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)

    I = np.identity(y_len)

    #Prior mean \mu
    mu_t = np.zeros((N_t, K))  #needs to be able to be added to each ensemble member

    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((y_len, K))
        for j in range(0, K): 
            g[:,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
        g_bar = np.mean(g, axis = 1)

        #Calculating perturbation matrices
        Cu = (u[i] - u_bar[:,np.newaxis])/np.sqrt(K - 1)   #X
        Cg = (g - g_bar[:,np.newaxis])/np.sqrt(K - 1)      #Y

        Cu_in = matrix_pseudo_inv(Cu)
        M = Cg@Cu_in

        #Add noise to \mu
        mu_0 = mu_t + (np.sqrt(2/alpha))*np.random.normal(0, 1, size = (N_t, K))
        #Update for IEKF w/ SL
        update = M.T@solve(M@M.T + I, 
                                            ((I*np.sqrt(2/alpha))@np.random.normal(0, 1, size = (y_len, K)) - 
                                            g) + Rsq_inv@y[:,np.newaxis] - M@(mu_0 - u[i]))  

        u[i + 1] = (1-alpha)*u[i] + alpha*(mu_0 + update)

        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args) ))/np.sqrt(y_len)
            #print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse: #Convergence criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)): #Convergence criteria
                exit = 2
                break
        else: 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence criteria
                exit = 3
                break

    return Bsq@u[i + 1] + mu[:,np.newaxis], (i+1)*K, exit 

def GN_CO(func, u_0, args, y, R, mu, B, min_rmse = 1, tol_x = 0.0001, tol_f = 0.0001, 
         method = "all", max_iter = 10e2):
    '''
    Gauss-Newton method from the Chen and Oliver 2013 paper (a.k.a. Iterative ensemble Kalman filter (IEKF))

    func: function 
        forward model function that is compared to the data
    u_0: list of floats 2D array
        array of intial ensemble members to be updated throught the algorithm (n,k)
        n -> number of parameters 
        k -> number of ensemble members
    args: tuple of variables?
        tuple of additional variables that are passed into the function
    y: List of floats array
        data vector with dimensions (m, 1)
    R: List of floats 2D array
        data covariance matrix with dimensions (m, m)
    mu: List of floats array
        prior mean with dimensions (n, 1)
    B: list of floats 2D array
        prior covariance matrix with dimensions (n,n)
    min_rmse: float
        target RMSE value
    tol_x: float
        target tolerance between x_n and x_(n+1)
    tol_f: float
        target tolerance between f_n and f_(n+1)
    method: String 
        method of convergence
        choose between ('all', 'rmse', 'tol_x', 'tol_f')
    max_iter: int
        maximum number of iterations before the algorithm stops
    
    Returns 
    -------
    u[i], i*K, exit
    
    u[i]: list of floats 2D array
        ensemble at final iteration
    (i+1)*K: float
        number of foward runs
    exit: int 
        if method = "all", exit criteria is returned 
        0 - hit max number of iterations
        1 - hit rmse tol
        2 - hit tol_x 
        3 - hit tol_f
    ''' 
    exit = 0 # exit criteria 
    
    (N_t, K) = u_0.shape  #parameter space size, ensemble size
    y_len = len(y)
    u = np.zeros((max_iter + 1, N_t, K))    #initialize parameter ensemble
    u[0] = u_0

    Rsq_inv = matrix_inv_sqrt(R)
    Bsq = matrix_sqrt(B)

    I = np.identity(y_len)

    #Prior mean \mu
    mu_t = np.zeros((N_t, K))  #needs to be able to be added to each ensemble member

    u_bar = np.mean(u[0], axis = 1)
    RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)

    #Algorithm:
    for i in range(0,max_iter): 
        #ensemble model output
        g = np.zeros((y_len, K))
        for j in range(0, K): 
            g[:,j] = Rsq_inv@func(Bsq@u[i,:,j] + mu, *args)
        g_bar = np.mean(g, axis = 1)

        #Calculating perturbation matrices
        Cu = (u[i] - u_bar[:,np.newaxis])/np.sqrt(K - 1)   #X
        Cg = (g - g_bar[:,np.newaxis])/np.sqrt(K - 1)      #Y

        Cu_in = matrix_pseudo_inv(Cu)
        M = Cg@Cu_in

        #Add noise to \mu
        mu_0 = mu_t + np.random.normal(0, 1, size = (N_t, K))
        #Update for IEKF w/ SL
        update = M.T@solve(M@M.T + I, 
                                            (np.random.normal(0, 1, size = (y_len, K)) - 
                                            g) + Rsq_inv@y[:,np.newaxis] - M@(mu_0 - u[i]))  

        u[i + 1] = mu_0 + update

        u_bar_1 = u_bar
        u_bar = np.mean(u[i + 1], axis = 1)

        if method == "all": 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse:  #Convergence Criteria
                exit = 1
                break
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)):  #Convergence Criteria
                exit = 2
                break
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            if func_diff < tol_f: #Convergence Criteria
                exit = 3
                break
        elif method == "rmse":
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args) ))/np.sqrt(y_len)
            #print('RMSE: %s Iter: %s' % (np.round(RMSE_u, 5), i + 1))
            if RMSE_u < min_rmse: #Convergence criteria
                exit = 1
                break
        elif method == "tol_x": 
            mean_diff = norm(u_bar - u_bar_1)/norm(u_bar_1)
            #print('Mean Difference: %s Iter: %s' % (np.round(mean_diff, 5), i + 1))
            if ((mean_diff < tol_x) and (norm(u_bar_1) > 1e-5) and (i > 0)): #Convergence criteria
                exit = 2
                break
        else: 
            RMSE_u_1 = RMSE_u
            RMSE_u = norm(Rsq_inv@(y - func(Bsq@u_bar + mu, *args)))/np.sqrt(y_len)
            func_diff = np.abs((RMSE_u - RMSE_u_1)/RMSE_u_1)
            #print('Function Difference: %s Iter: %s' % (np.round(func_diff, 5), i + 1))
            if func_diff < tol_f: #Convergence criteria
                exit = 3
                break

    return Bsq@u[i + 1] + mu[:,np.newaxis], (i+1)*K, exit 


