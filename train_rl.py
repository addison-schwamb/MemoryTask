'''
Functions to train RNN with reinforcement learning, test RNN, save data
Written in Python 3.10.11
@ Addison
'''

import time
import pickle
import numpy as np
from scipy import sparse, stats
from matplotlib import pyplot as plt

def initialize_base_net(params, learning, dist='Gauss'):
    ''' Function to initialize network states and initial weights
    Args:
    params: dictionary containing all parameters for network setup, task, and training
    learning: learning method (currently 'PG' for policy gradient is the only one supported
    dist: distribution for initialization of weights and states -- can be either 'Gauss' or 'Uniform'
    '''
    net_prs = params['network']
    train_prs = params['train']
    msc_prs = params['msc']
    N = net_prs['N']
    rng = np.random.RandomState(msc_prs['seed'])
    std = 1/np.sqrt(net_prs['pg'] * N)
    
    W = std * sparse.random(N, N, density=net_prs['pg'], random_state=msc_prs['seed'], data_rvs=rng.randn).toarray()
    if dist == 'Gauss':
        x = 0.01 * rng.normal(size=(N,1))
        wi = (0.1 * rng.normal(size=(N, net_prs['d_input']))) / net_prs['input_var']
        wf = (0.1 * rng.normal(size=(N, net_prs['d_output']))) / net_prs['fb_var']
        wd = 0.1 * rng.normal(size=(N, net_prs['d_dummy']))
        wo = 0.01 * rng.normal(size=(N, net_prs['d_output']))
    elif dist == 'Uniform':
        print('initialization is uniform')
        x = 0.001 * (2 * rng.random(N, 1) - 1)
        wi = (0.02 * rng.random(N, net_prs['d_input']) - 0.01) / net_prs['input_var']
        wf = (0.02 * rng.random(N, net_prs['d_output']) - 0.01) / net_prs['fb_var']
        wd = 0.02 * rng.random(N, net_prs['d_dummy']) - 0.01
        wo = 0.002 * rng.random(N, net_prs['d_output']) - 0.001
    
    if learning == 'PG':
        sigma2 = train_prs['sigma2']
        Sigma = np.diagflat(sigma2 * np.ones([N, 1]))
        net = {'Sigma': Sigma, 'W': W, 'wi': wi, 'wf': wf, 'wd': wd, 'wo': wo}
    
    return net, x

def zero_fat_mats(params, trial_lens, is_train=True):
    ''' Initialize zero matrices for recording values across training/testing
    Args:
    params: dictionary of params for network setup, task, and training
    trial_lens: list of the number of timesteps in each trial
    is_train: flag for setting matrix length to number of training trials or testing trials
    '''
    net_prs = params['network']
    train_prs = params['train']
    if is_train:
        total_size = train_prs['n_train']
        total_steps = int(sum(trial_lens[0:train_prs['n_train']]))
    else:
        total_size = train_prs['n_test']
        total_steps = int(sum(trial_lens[train_prs['n_train']:]))
    
    z_mat = np.zeros([net_prs['d_output'], total_steps])
    x_mat = np.zeros([net_prs['N'], total_steps])
    r_mat = np.zeros([net_prs['N'], total_steps])
    eps_mat = np.zeros([net_prs['N'], total_steps])
    rwd_mat = np.zeros([total_size,])
    err_mat = np.zeros([total_size,])
    
    return z_mat, x_mat, r_mat, eps_mat, rwd_mat, err_mat

def train_pg(params, exp_mat, target_mat, positions, trial_lens, dist='Gauss'):
    ''' Function to implement training using episodic policy gradient
    Args:
    params: dictionary of params for network setup, task, and training
    exp_mat: input matrix for the network
    positions: list of targets to remember
    trial_lens: list of the number of timesteps in each trial
    dist: initial distribution for neural network
    Returns:
    x_train: current state of trained network
    params: params for trained network
    err_mat: matrix of error for each trial
    '''
    tic = time.time()
    
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    rng = np.random.default_rng(msc_prs['seed'])
    
    N, dt, tau = net_prs['N'], net_prs['dt'], net_prs['tau']
    g, pg = net_prs['g'], net_prs['pg']
    alpha0, sigma20, max_grad = train_prs['alpha'], train_prs['sigma2'], train_prs['max_grad']
    var_smooth, tau_H, tau_phi = train_prs['var_smooth'], train_prs['tau_H'], train_prs['tau_phi']
    tau_alpha, tau_sigma = train_prs['tau_alpha'], train_prs['tau_sigma']
    print('Policy gradient params: ', alpha0, '_', sigma20, '_', max_grad, '_', tau_H, '_', tau_phi,
            '_', tau_alpha, '_', tau_sigma)
    
    net, x0 = initialize_base_net(params, 'PG', dist=dist)
    x = x0
    Sigma, W, wi, wf, wd, wo = net['Sigma'], net['W'], net['wi'], net['wf'], net['wd'], net['wo']
    sigma2 = sigma20
    alpha = alpha0
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)
    eps = np.matmul(Sigma, rng.normal(size=(N, 1)))
    
    z_mat, x_mat, r_mat, eps_mat, rwd_mat, err_mat = zero_fat_mats(params, trial_lens, is_train=True)
    update_step = train_prs['update_step']
    train_steps = int(sum(trial_lens[0:train_prs['n_train']]))
    response_steps = int(task_prs['time_intervals']['response'] / dt)
    trial = 0
    trial_i = 0
    last_stop = 0
    R_fix, R_res = 0, 0
    err_fix = 0
    phi_s = 0
    phi_l = 0
    phi = 0
    
    for i in range(train_steps):
        trial_steps = trial_lens[trial]
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        z_mat[:, i] = z.reshape(-1)
        eps_mat[:, i] = eps.reshape(-1)
        
        dx = -x + g * np.matmul(W, r) + np.matmul(wi, exp_mat[:,i].reshape([net_prs['d_input'], 1])) +\
            np.matmul(wf, z.reshape([net_prs['d_output'], 1]))
        x = x + (dx * dt) / tau
        x_eps = x + eps
        r = np.tanh(x)
        r_eps = np.tanh(x_eps)
        z = np.matmul(wo.T, r_eps)
        zd = np.matmul(wd.T, r)
        
        if np.all(exp_mat[:,i] == 0.5):
            R_fix = -1 * np.matmul((exp_mat[:,i] - z.reshape(-1)).reshape([1,2]), (exp_mat[:,i] - z.reshape(-1)).reshape([2,1]))
            err_fix = np.linalg.norm(exp_mat[:,i] - z.reshape(-1))
        
        if np.any(target_mat[:, i] != 0.):
            if (trial_i) % update_step == 0 and trial_i >= update_step:
                eps_r = np.zeros([N, N])
                eps_in = np.zeros([N, net_prs['d_input']])
                eps_fb = np.zeros([N, net_prs['d_output']])
                
                R_res = -1 * np.matmul((positions[:,trial] - z.reshape(-1)).reshape([1,net_prs['d_output']]), (positions[:,trial] - z.reshape(-1)).reshape([net_prs['d_output'],1]))
                err_res = np.linalg.norm(positions[:,trial] - z.reshape(-1))
                R = R_fix + R_res
                rwd_mat[trial] = R
                err_mat[trial] = err_fix + err_res
                steps_since_update = i - last_stop
                
                for j in range(1, steps_since_update+1):
                    idx = int(i - steps_since_update + j)
                    r_cum = 0
                    input_cum = 0
                    output_cum = 0
                    for k in range(1, j+1):
                        r_cum += ((1 - dt) ** (k - 1)) * dt * r_mat[:, idx-k]
                        input_cum += ((1 - dt) ** (k - 1)) * dt * exp_mat[:, idx-k]
                        output_cum += ((1 - dt) ** (k - 1)) * dt * z_mat[:, idx-k]
                    eps_r += np.outer(eps_mat[:, idx], r_cum)
                    eps_in += np.outer(eps_mat[:, idx], input_cum)
                    eps_fb += np.outer(eps_mat[:, idx], output_cum)
                
                deltaW = (alpha * dt / sigma2) * R * eps_r
                if np.linalg.norm(deltaW, ord='fro') > max_grad:
                    deltaW = (max_grad / np.linalg.norm(deltaW, ord='fro')) * deltaW
                deltawi = (alpha * dt / sigma2) * R * eps_in
                deltawf = (alpha * dt / sigma2) * R * eps_fb
                
                H = np.exp((-np.var(rwd_mat[max(0,trial-var_smooth):trial+1]) - abs(R)) / tau_H) + 0.5 * np.log(2 * np.pi * np.e)
                phi_s = (1 - 1/tau_phi) * phi_s + (1/tau_phi) * np.exp(-H)
                phi_l = (1 - 1/tau_phi) * phi_l + (1/tau_phi) * phi_s
                phi += phi_l
                
                sigma2 = sigma20 * np.exp(-phi/tau_sigma)
                alpha = sigma2 * (alpha0 / sigma20) * np.exp(-phi/tau_alpha)
                Sigma = np.diagflat(sigma2 * np.ones([N, 1]))
                W += deltaW
                wi += deltawi
                wf += deltawf
                
                last_stop = i
            if trial_i >= trial_steps:
                trial_i = 0
                x = x0
                trial += 1
        
        eps = np.matmul(Sigma, rng.normal(size=(N,1)))
        trial_i += 1
    
    toc = time.time()
    print('train time = ', (toc - tic)/60)
    print('Final Reward: ',rwd_mat[trial])
    print('Fixate Error: ', err_fix)
    print('Response Error: ',err_res)
    plt.figure()
    plt.plot(range(trial), rwd_mat[0:trial])
    plt.xlabel('Trial Number')
    plt.ylabel('Reward')
    plt.title('Motor Task Learning')
    plt.savefig(msc_prs['name'] + '_trial' + str(trial+1) + '_random_delay.png')
    plt.show()
    
    plt.figure()
    plt.plot(z_mat[0,i-response_steps:i], z_mat[1, i-response_steps:i])
    plt.scatter(positions[0,trial], positions[1,trial])
    #plt.xlabel('x position')
    #plt.ylabel('y position')
    plt.title('Trajectory')
    
    plt.show()
    model_params = {'W': W, 'wo': wo, 'wi': wi, 'wf': wf, 'wd': wd, 'Sigma': Sigma}
    params['model'] = model_params
    task_prs['counter'] = i
    
    return x, params, err_mat