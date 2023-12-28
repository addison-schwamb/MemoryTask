import argparse
import json
import sys
import pickle
from memory_task import *
from train_rl import *
dir = ''

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs = args.d

def set_all_parameters(g, pg, fb_var, alpha, sigma, var_smooth, n_train, radius, seed, init_dist, train_method, activation='tanh'):
    params = dict()
    
    net_params = dict()
    net_params['d_input'] = 2
    net_params['d_output'] = 2
    net_params['d_dummy'] = 2
    net_params['tau'] = 1
    net_params['dt'] = 0.1
    net_params['g'] = g
    net_params['pg'] = pg
    net_params['N'] = 1000
    net_params['input_var'] = 50.0
    net_params['fb_var'] = fb_var
    params['network'] = net_params
    
    task_params = dict()
    t_intervals = dict()
    t_intervals['fixate_on_init'], t_intervals['fixate_off_init'] = 0, 0
    t_intervals['fixate_on_postdelay'], t_intervals['fixate_off_postdelay'] = 5, 0
    t_intervals['cue_on'], t_intervals['cue_off'] = 5, 0
    t_intervals['delay_max'] = 10
    t_intervals['response'] = 0.2
    task_params['time_intervals'] = t_intervals
    task_params['radius'] = radius
    params['task'] = task_params
    
    train_params = dict()
    train_params['update_step'] = 2
    train_params['n_train'] = n_train
    train_params['n_test'] = 100
    train_params['init_dist'] = init_dist
    train_params['activation'] = activation
    if train_method == 'PG':
        train_params['alpha'] = alpha
        train_params['sigma2'] = sigma
        train_params['max_grad'] = 0.05
        train_params['tau_H'] = 2500
        train_params['tau_phi'] = 3500
        train_params['tau_alpha'] = 9
        train_params['tau_sigma'] = 24
        train_params['var_smooth'] = var_smooth
    params['train'] = train_params
    
    other_params = dict()
    other_params['name'] = '_'.join(['{}'.format(val) if type(val) != list
                                    else '{}'.format(''.join([str(s) for s in val])) for k, val in kwargs.items()])
    other_params['seed'] = seed
    other_params['train_method'] = train_method
    params['msc'] = other_params
    print('name is: ', other_params['name'])
    
    return params

params = set_all_parameters(**kwargs)
task_prs = params['task']
train_prs = params['train']
net_prs = params['network']
msc_prs = params['msc']

task = memory_task_experiment(train_prs['n_train'], train_prs['n_test'], task_prs['time_intervals'], net_prs['dt'],
                                task_prs['radius'], msc_prs['seed'], net_prs['d_input'], net_prs['d_output'])
exp_mat, target_mat, positions, trial_lens = task.experiment()

if msc_prs['train_method'] == 'PG':
    print('Training with episodic policy gradient')
    x_train, params, err_mat_train = train_pg(params, exp_mat, target_mat, positions, trial_lens, dist=train_prs['init_dist'])