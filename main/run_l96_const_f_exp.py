'''
Finding optimal cost of each EKI algorith for the stochastic l96 problem estimating scalar gamma

By: Rebecca Gjini
'''
#Import statements
from functools import partial
import numpy as np
from scipy.optimize import least_squares
import multiprocessing as mp
import EnsembleKalmanAlgorithms as EKA
import main.l96.L96_const_model as gl96

path = "l96/const/"

def model_setup_stats():
    t = 0.01
    T = 14
    nx = 1
    x0 = np.loadtxt(path + 'data/x0.txt', delimiter = ',') 
    ic_cov_sqrt = np.loadtxt(path + 'data/ic_cov_sqrt.txt', delimiter = ',') #sampling matrix
    mu = np.array([np.loadtxt(path + 'data/mu.txt', delimiter = ',')])
    B = np.array([np.loadtxt(path + 'data/B.txt', delimiter = ',')])
    R= np.loadtxt(path + 'data/R.txt', delimiter = ',')
    y = np.loadtxt(path + 'data/y.txt', delimiter = ',')

    return t, T, nx, x0, ic_cov_sqrt, mu, B, R, y


def run_teki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    Ne_vals = np.arange(2, 30, 2)  # list of ensemble sizes to test
    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/teki_ens_sizes.txt'), Ne_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for Ne in Ne_vals: 
        #Intitializing EKI ensemble 
        max_runs = 100   #set a maximum number of runs 
        N_t = nx         

        u = np.random.normal(0, 1, size = (N_t,Ne))   #initialize parameter ensemble

        #TEKI Test 
        u_out, f_out, _ = EKA.TEKI(gl96.G, u, (x0, t, T, ic_cov_sqrt), 
                          y, R, mu, B, min_rmse = rmse, method = method_type, 
                             tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)

    np.savetxt((path + 'runs/TEKI/'+method_type+str(rmse)+'/fow_runs/teki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/TEKI/'+method_type+str(rmse)+'/ensemble/teki_%d.npz') %ran, *u_all)

def run_etki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    Ne_vals = np.arange(2, 30, 2)  # list of ensemble sizes to test
    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/etki_ens_sizes.txt'), Ne_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for Ne in Ne_vals: 
        max_runs = 100   #set a maximum number of runs 
        N_t = nx         

        u = np.random.normal(0, 1, size = (N_t,Ne))    #initialize parameter ensemble

        #ETKI Test 
        u_out, f_out, _ = EKA.ETKI(gl96.G, u, (x0, t, T, ic_cov_sqrt), 
                          y, R, mu, B, min_rmse = rmse, method = method_type, 
                             tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)

    np.savetxt((path + 'runs/ETKI/'+method_type+str(rmse)+'/fow_runs/etki_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/ETKI/'+method_type+str(rmse)+'/ensemble/etki_%d.npz') %ran, *u_all)

def run_iekf(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    Ne_vals = np.arange(2, 30, 2) # list of ensemble sizes to test
    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/iekf_ens_sizes.txt'), Ne_vals, delimiter = ',')
    u_all = []
    fow_runs = []


    for Ne in Ne_vals: 
        #Intitializing EKI ensemble 
        max_runs = 100   #set a maximum number of runs 
        N_t = nx         
        alpha = 1     #adjustment factor

        u =  np.random.normal(0, 1, size = (N_t,Ne))     #initialize parameter ensemble

        #IEFK Test 
        u_out, f_out, _ = EKA.IEKF(gl96.G, u, (x0, t, T, ic_cov_sqrt), 
                          y, R, mu, B, min_rmse = rmse, method = method_type, 
                             tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)


    np.savetxt((path + 'runs/IEKF/'+method_type+str(rmse)+'/fow_runs/iekf_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/IEKF/'+method_type+str(rmse)+'/ensemble/iekf_%d.npz') %ran, *u_all)

def run_gnco(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    Ne_vals = np.arange(2, 30, 2) # list of ensemble sizes to test
    np.savetxt((path + 'runs/GNCO/'+method_type+str(rmse)+'/gnco_ens_sizes.txt'), Ne_vals, delimiter = ',')
    u_all = []
    fow_runs = []
    #rmse = []


    for Ne in Ne_vals: 
        #Intitializing EKI ensemble 
        max_runs = 100   #set a maximum number of runs 
        N_t = nx         
        alpha = 1     #adjustment factor

        u =  np.random.normal(0, 1, size = (N_t,Ne))     #initialize parameter ensemble

        #GNCO Test 
        u_out, f_out, _ = EKA.GN_CO(gl96.G, u, (x0, t, T, ic_cov_sqrt), 
                          y, R, mu, B, min_rmse = rmse, method = method_type, 
                             tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)
        fow_runs.append(f_out)
        u_all.append(u_out)


    np.savetxt((path + 'runs/GNCO/'+method_type+str(rmse)+'/fow_runs/gnco_%d.txt') %ran, fow_runs, delimiter = ',')
    np.savez((path + 'runs/GNCO/'+method_type+str(rmse)+'/ensemble/gnco_%d.npz') %ran, *u_all)

def run_uki(method_type, rmse, ran):
    np.random.seed(ran)  #initialize random seed                     

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    #Intitializing EKI ensemble 
    max_runs = 100   #set a maximum number of runs 
    a = 1           #adjustment factor

    #UKI Test 
    u_out, f_out, _ = EKA.UKI(gl96.G, (x0, t, T, ic_cov_sqrt), 
                          y, R, mu, B, min_rmse = rmse, method = method_type, 
                             tol_x = 1e-4, tol_f = 1e-4, max_iter = max_runs)


    np.savetxt((path + 'runs/UKI/'+method_type+str(rmse)+'/fow_runs/uki_%d.txt') %ran, [f_out], delimiter = ',')
    np.savez((path + 'runs/UKI/'+method_type+str(rmse)+'/ensemble/uki_%d.npz') %ran, *u_out)

def run_lmfd(ran):
    np.random.seed(ran)  #initialize random seed

    t, T, nx, x0, ic_cov_sqrt, mu, B, R, y  = model_setup_stats()

    R_sqrt_in = EKA.matrix_inv_sqrt(R)
    B_sqrt = EKA.matrix_sqrt(B)

    N_t = nx 

    u = np.random.normal(0, 1, size = N_t)    #initialize parameter guess

    #LMFD Test 
    solution = least_squares(gl96.r, u, args=(x0, t, T, ic_cov_sqrt, y, R_sqrt_in, mu, B_sqrt), method = 'lm', 
                         xtol = 1e-8, ftol=1e-08)

    np.savetxt((path + 'runs/LMFD/fow_runs/lmfd_%d.txt')  %ran, [solution.nfev], delimiter = ',')
    np.savez((path + 'runs/LMFD/solution/lmfd_%d.npz') %ran, *(B_sqrt@solution.x + mu))
    np.savez((path + 'runs/LMFD/init_condition/lmfd_%d.npz') %ran, *(B_sqrt@u + mu))


def multirun(func, num_runs_start, num_runs_end, fargs=()): 
    partial_worker = partial(func, *fargs)
    with mp.Pool(processes=16) as pool:
        # Use map to distribute tasks
        results = pool.map(partial_worker, range(num_runs_start, num_runs_end + 1))
