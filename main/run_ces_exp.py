'''
Finding optimal cost of each EKI algorith for l63 problem from the calibrate emulate sample paper

By: Rebecca Gjini
'''
#Import statements
from functools import partial
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import least_squares
import multiprocessing as mp
import EnsembleKalmanAlgorithms as EKA
import l63.L63_model as l63

path = 'l63/ces/'

def model_setup():
    og_sigma = 10.0  
    #og_rho = 28.0 #ln(28) = 3.33 
    #og_beta = 8.0 / 3.0 #ln(2.66) = 0.98

    time_step = 0.01  #time step
    T = 40  #total time

    x0 = np.loadtxt(path + 'data/x0.txt', delimiter = ',') #intital condition used for the data

    ic_cov_sqrt = np.loadtxt(path + 'data/ic_cov_sqrt.txt', delimiter = ',') #sqrt of covariance of noise that will be added to x0

    y = np.loadtxt(path + 'data/y.txt', delimiter = ',')
    R = np.loadtxt(path + 'data/R.txt', delimiter = ',')
    mu = np.loadtxt(path + 'data/mu.txt', delimiter = ',')
    B = np.loadtxt(path + 'data/B.txt', delimiter = ',')

    return og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B



def run_teki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(2,51, 1)
    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/teki_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 200   #set a maximum number of runs 
        N_t = 2         #we only estimate beta and rho
        a = 1           #adjustment factor
        alpha = 1     #adjustment factor

        u = np.zeros((N_t, K))    #initialize parameter ensemble
        u[0,:] = np.random.normal(0, 1, size = K)
        u[1,:] = np.random.normal(0, 1, size = K)

        #TEKI Test 
        u_out, f_out, _ = EKA.TEKI(l63.G, u, (og_sigma, x0, ic_cov_sqrt, time_step, T), 
                            y, R, mu, B, method = method_type, 
                            min_rmse = rmse, tol_x = 0.001, tol_f = 0.001, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)


    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/fow_runs/teki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/TEKI/'+method_type+str(rmse)+'/ensemble/teki_%d.npz') %ran, *u_all)

def run_etki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(2,51, 1)
    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/etki_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []
    #rmse = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 200   #set a maximum number of runs 
        N_t = 2         #we only estimate beta and rho
        a = 1           #adjustment factor
        alpha = 1     #adjustment factor

        u = np.zeros((N_t, K))    #initialize parameter ensemble
        u[0,:] = np.random.normal(0, 1, size = K)
        u[1,:] = np.random.normal(0, 1, size = K)

        #ETKI Test 
        u_out, f_out, _ = EKA.ETKI(l63.G, u, (og_sigma, x0, ic_cov_sqrt, time_step, T), 
                            y, R, mu, B, method = method_type, 
                            min_rmse = rmse, tol_x = 0.001, tol_f = 0.001, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)

    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/fow_runs/etki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/ETKI/'+method_type+str(rmse)+'/ensemble/etki_%d.npz') %ran, *u_all)

def run_iekf(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(3,51, 1) #cannot handle 2 ensemble members 
    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/iekf_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 200   #set a maximum number of runs 
        N_t = 2         #we only estimate beta and rho
        a = 1           #adjustment factor
        alpha = 1     #adjustment factor

        u = np.zeros((N_t, K))    #initialize parameter ensemble
        u[0,:] = np.random.normal(0, 1, size = K)
        u[1,:] = np.random.normal(0, 1, size = K)

        #IEKF Test 
        u_out, f_out, _ = EKA.IEKF(l63.G, u, (og_sigma, x0, ic_cov_sqrt, time_step, T), 
                            y, R, mu, B, method = method_type, 
                            min_rmse = rmse, tol_x = 0.001, tol_f = 0.001, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)


    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/fow_runs/iekf_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/IEKF/'+method_type+str(rmse)+'/ensemble/iekf_%d.npz') %ran, *u_all)

def run_gnco(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(3,51, 1) #cannot handle 2 ensemble members 
    np.savetxt((path + 'runs/GNCO/'+method_type+str(rmse)+'/gnco_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 200   #set a maximum number of runs 
        N_t = 2         #we only estimate beta and rho
        a = 1           #adjustment factor
        alpha = 1     #adjustment factor

        u = np.zeros((N_t, K))    #initialize parameter ensemble
        u[0,:] = np.random.normal(0, 1, size = K)
        u[1,:] = np.random.normal(0, 1, size = K)

        #GNCO Test 
        try: 
            u_out, f_out, _ = EKA.GN_CO(l63.G, u, (og_sigma, x0, ic_cov_sqrt, time_step, T), 
                                y, R, mu, B, method = method_type, 
                                min_rmse = rmse, tol_x = 0.001, tol_f = 0.001, max_iter = max_runs)
        except LinAlgError as a: 
            u_out = np.empty((N_t, K))
            u_out[:] = np.nan
            f_out = max_runs*K
        fow_runs.append(f_out)
        u_all.append(u_out)


    np.savetxt((path + 'runs/GNCO/'+method_type+str(rmse)+'/fow_runs/gnco_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/GNCO/'+method_type+str(rmse)+'/ensemble/gnco_%d.npz') %ran, *u_all)

def run_uki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    #Intitializing EKI ensemble 
    max_runs = 200   #set a maximum number of runs 
    a = 1           #adjustment factor
    alpha = 1     #adjustment factor

    #UKI Test 
    u_out, f_out, _ = EKA.UKI(l63.G, (og_sigma, x0, ic_cov_sqrt, time_step, T), 
                        y, R, mu, B, method = method_type, 
                        min_rmse = rmse, tol_x = 0.001, tol_f = 0.001, max_iter = max_runs)


    np.savetxt((path + 'runs/UKI/'+method_type+str(rmse)+'/fow_runs/uki_%d.txt') %ran, [f_out], delimiter = ',')
    np.savez((path + 'runs/UKI/'+method_type+str(rmse)+'/ensemble/uki_%d.npz') %ran, *u_out)

def run_lmfd(ran):
    np.random.seed(ran)  #initialize random seed

    og_sigma, time_step, T, x0, ic_cov_sqrt, y, R, mu, B = model_setup()

    R_sqrt_in = EKA.matrix_inv_sqrt(R)
    B_sqrt = EKA.matrix_sqrt(B)

    N_t = 2 #we only estimate beta and rho

    u = np.zeros(N_t)    #initialize parameter ensemble
    u[0] = np.random.normal(0, 1, size = 1)
    u[1] = np.random.normal(0, 1, size = 1)

    #LMFD Test 
    solution = least_squares(l63.r, u, args=(og_sigma, x0, ic_cov_sqrt, time_step, 
                                          T, y, R_sqrt_in, mu, B_sqrt), method = 'lm', 
                                          xtol = 1e-8, ftol = 1e-8)

    np.savetxt((path + 'runs/LMFD/fow_runs/lmfd_%d.txt')  %ran, [solution.nfev], delimiter = ',')
    np.savez((path + 'runs/LMFD/solution/lmfd_%d.npz') %ran, *(B_sqrt@solution.x + mu))
    np.savez((path + 'runs/LMFD/init_condition/lmfd_%d.npz') %ran, *(B_sqrt@u + mu))


def multirun(func, num_runs_start, num_runs_end, fargs=()): 
    partial_worker = partial(func, *fargs)
    with mp.Pool(processes=16) as pool:
        # Use map to distribute tasks
        results = pool.map(partial_worker, range(num_runs_start, num_runs_end + 1))

