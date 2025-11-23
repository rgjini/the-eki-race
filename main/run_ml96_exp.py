'''
Finding optimal cost of each EKI algorith for the stochastic modified l96 problem

By: Rebecca Gjini

'''
#Import statements
from functools import partial
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import least_squares
import multiprocessing as mp
import EnsembleKalmanAlgorithms as EKA
import main.l96.mL96_model as ml96

path = "l96/mp/"

def model_setup():
    nx = 40
    # gamma = 8 + 6*np.sin((4*np.pi*np.arange(0, nx, 1))/nx)
    t = 0.01 
    T = 54

    ic_cov_sqrt = np.loadtxt(path + 'data/ic_cov_sqrt.txt', delimiter = ',') #sampling matrix

    x0 = np.loadtxt(path + 'data/x0.txt', delimiter = ',') 
    y = np.loadtxt(path + 'data/y.txt', delimiter = ',')
    R = np.loadtxt(path + 'data/R.txt', delimiter = ',')
    mu = np.loadtxt(path + 'data/mu.txt', delimiter = ',')
    B = np.loadtxt(path + 'data/B.txt', delimiter = ',')

    return nx, x0, t, T, ic_cov_sqrt, y, R, mu, B


def run_teki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(60, 200, 5)
    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/teki_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 150   #set a maximum number of runs 
        N_t = nx         

        u = np.random.normal(0, 1, size = (N_t,K))   #initialize parameter ensemble

        #TEKI Test 
        u_out, f_out, _ = EKA.TEKI(ml96.G, u, (x0, t, T, ic_cov_sqrt),  
                            y, R, mu, B, method = method_type, 
                            min_rmse = rmse, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)

    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/fow_runs/teki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/TEKI/'+method_type+str(rmse)+'/ensemble/teki_%d.npz') %ran, *u_all)

def run_etki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(45, 200, 5)
    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/etki_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 150   #set a maximum number of runs 
        N_t = nx         

        u = np.random.normal(0, 1, size = (N_t,K))    #initialize parameter ensemble

        #ETKI Test 
        u_out, f_out, _ = EKA.ETKI(ml96.G, u, (x0, t, T, ic_cov_sqrt), 
                            y, R, mu, B, method = method_type, 
                            min_rmse = rmse, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)

    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/fow_runs/etki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/ETKI/'+method_type+str(rmse)+'/ensemble/etki_%d.npz') %ran, *u_all)

def run_iekf(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(45, 200, 5)
    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/iekf_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []
    #rmse = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 150   #set a maximum number of runs 
        N_t = nx        
        alpha = 1     #adjustment factor

        u =  np.random.normal(0, 1, size = (N_t,K))     #initialize parameter ensemble

        #IEKF Test 
        u_out, f_out, _ = EKA.IEKF(ml96.G, u, (x0, t, T, ic_cov_sqrt), 
                            y, R, mu, B, alpha = alpha, method = method_type, 
                            min_rmse = rmse, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)

    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/fow_runs/iekf_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/IEKF/'+method_type+str(rmse)+'/ensemble/iekf_%d.npz') %ran, *u_all)

def run_gnco(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    K_vals = np.arange(45, 200, 5)
    np.savetxt((path + 'runs/GNCO/'+method_type+str(rmse)+'/gnco_ens_sizes.txt'), K_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for ii in K_vals: 
        #Intitializing EKI ensemble 
        K = ii         #number of ensemble members
        max_runs = 150   #set a maximum number of runs 
        N_t = nx         
        alpha = 1     #adjustment factor

        u =  np.random.normal(0, 1, size = (N_t,K))     #initialize parameter ensemble

        #GNCO Test 
        try: 
            u_out, f_out, _ = EKA.GN_CO(ml96.G, u, (x0, t, T, ic_cov_sqrt), 
                                y, R, mu, B, method = method_type, 
                                min_rmse = rmse, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
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

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    #Intitializing EKI ensemble 
    max_runs = 150   #set a maximum number of runs 
    a = 1           #adjustment factor

    #UKI Test 
    u_out, f_out, _ = EKA.UKI(ml96.G, (x0, t, T, ic_cov_sqrt), 
                        y, R, mu, B, method = method_type, 
                        min_rmse = rmse, tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)

    np.savetxt((path + 'runs/UKI/'+method_type+str(rmse)+'/fow_runs/uki_%d.txt') %ran, [f_out], delimiter = ',')
    np.savez((path + 'runs/UKI/'+method_type+str(rmse)+'/ensemble/uki_%d.npz') %ran, *u_out)

def run_lmfd(ran):
    np.random.seed(ran)  #initialize random seed

    nx, x0, t, T, ic_cov_sqrt, y, R, mu, B = model_setup()

    R_sqrt_in = EKA.matrix_inv_sqrt(R)
    B_sqrt = EKA.matrix_sqrt(B)

    N_t = nx 

    u = np.random.normal(0, 1, size = N_t)    #initialize parameter guess

    #LMFD Test 
    solution = least_squares(ml96.r, u, args=(x0, t, T, ic_cov_sqrt, y, R_sqrt_in, mu, B_sqrt), method = 'lm', 
                                          xtol = 1e-8, ftol = 1e-8)

    np.savetxt((path + 'runs/LMFD/fow_runs/lmfd_%d.txt')  %ran, [solution.nfev], delimiter = ',')
    np.savez((path + 'runs/LMFD/solution/lmfd_%d.npz') %ran, *(B_sqrt@solution.x + mu))
    np.savez((path + 'runs/LMFD/init_condition/lmfd_%d.npz') %ran, *(B_sqrt@u + mu))


def multirun(func, num_runs_start, num_runs_end, fargs=()): 
    partial_worker = partial(func, *fargs)
    with mp.Pool(processes=16) as pool:
        # Use map to distribute tasks
        results = pool.map(partial_worker, range(num_runs_start, num_runs_end + 1))

