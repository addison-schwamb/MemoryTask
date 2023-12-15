'''class to contruct the experiment -- spatial memory task
written in Python 3.10.11
@ Addison
'''

import numpy as np
import random

class memory_task_experiment:

    def __init__(self, train_trials_size, test_trials_size, t_intervals, dt, radius, seed, input_dim, output_dim):
        ''' Creates spatial memory task object
        Args:
        train_trials_size: number of trials for training
        test_trials_size: number of trials for testing
        t_intervals: dictionary that contains task time intervals
        dt: Euler integration time step
        radius: radius of the circle the target points lie on
        seed: random seed
        '''
        self.train_size = train_trials_size
        self.test_size = test_trials_size
        self.trial_size = train_trials_size + test_trials_size
        self.t_intervals = t_intervals
        self.dt = dt
        self.radius = radius
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.fixate_value = 0.5
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def expand(self, on_intv, off_intv, input_dim, theta, input_type=None):
        ''' Takes the on and off interval of each input type and expands it by dt, and assigns values to each input type
        Args:
        on_intv: time to present the input value
        off_intv: time to have the input value not presented during the epoch
        input_dim: dimension of the input matrix
        input_type ('fixate', 'cue', 'delay', 'response'): defines which epoch of the task is being expanded
        theta: angle of target
        Returns:
        x_pos: x position for selected epoch
        y_pos: y position for selected epoch
        input_mat (input_dim x t_intv): input mat for selected epoch
        '''
        x_pos = None
        y_pos = None
        if input_type != 'delay':
            t_intv = (on_intv + off_intv) / self.dt
            input_mat = np.zeros([input_dim, int(t_intv)])
            
            if input_type == 'fixate':
                input_mat[:, 0:int(on_intv / self.dt)] = self.fixate_value
            elif input_type == 'cue':
                x_pos = self.radius * np.cos(theta)
                y_pos = self.radius * np.sin(theta)
                input_mat[:, 0:int(on_intv / self.dt)] = np.asarray([[x_pos], [y_pos]])
        else:
            delay_len = self.rng.integers(low=0, high=on_intv)
            t_intv = (delay_len + off_intv) / self.dt
            input_mat = np.zeros([input_dim, int(t_intv)])
        
        return x_pos, y_pos, input_mat
        
    def generate_trial(self, t_intv, theta):
        '''
        generates a single trial
        Return:
        x_pos: x target position
        y_pos: y target position
        exp_trial (input_dim x t_trial): the input matrix for the trial
        target_trial (output_dim x t_trial): the target matrix for the trial
        '''
        input_dim = self.input_dim
        output_dim = self.output_dim
        _, _, fixate_init = self.expand(t_intv['fixate_on_init'], t_intv['fixate_off_init'], input_dim, theta, input_type='fixate')
        
        x_pos, y_pos, cue = self.expand(t_intv['cue_on'], t_intv['cue_off'], input_dim, theta, input_type='cue')
        
        _, _, delay = self.expand(t_intv['delay_max'], 0, input_dim, theta, input_type='delay')
        
        _, _, fixate_postdelay = self.expand(t_intv['fixate_on_postdelay'], t_intv['fixate_off_postdelay'], input_dim, theta, input_type='fixate')
        
        _, _, response = self.expand(0, t_intv['response'], input_dim, theta, input_type='response')
        
        fix_cue = np.concatenate((fixate_init, cue), axis=1)
        fix_cue_delay = np.concatenate((fix_cue, delay), axis=1)
        fix_cue_delay_fix = np.concatenate((fix_cue_delay, fixate_postdelay), axis=1)
        trial = np.concatenate((fix_cue_delay_fix, response), axis=1)
        
        return x_pos, y_pos, trial
    
    def experiment(self):
        ''' Generates a sequence of trials equal to num_trials and generates target locations
        Returns:
        exp_mat (num_input x t_exp): input to network
        target_mat (num_output x t_exp): output for network to match
        positions (num_output x num_trials): list of positions to remember
        '''
        t_intv = self.t_intervals
        input_dim = self.input_dim
        output_dim = self.output_dim
        positions = np.zeros([output_dim, self.trial_size])
        trial_len_mat = np.zeros([self.trial_size,])
        pi = np.pi
        angles = [0]
        for i in range(0, self.trial_size):
            theta = self.rng.choice(angles)
            x_pos, y_pos, trial = self.generate_trial(t_intv, theta)
            t_trial = np.shape(trial)[1]
            if i == 0:
                exp_mat = trial
                target_mat = np.zeros([output_dim, t_trial])
                target_mat[:, -int(t_intv['response'] / self.dt):] = np.outer(np.asarray([[x_pos], [y_pos]]), np.ones([1, int(t_intv['response'] / self.dt)]))
            else:
                exp_mat = np.concatenate((exp_mat, trial), axis=1)
                target_mat = np.concatenate((target_mat, np.zeros([output_dim, t_trial])), axis=1)
                target_mat[:, -int(t_intv['response'] / self.dt):] = np.outer(np.asarray([[x_pos], [y_pos]]), np.ones([1, int(t_intv['response'] / self.dt)]))
            positions[:, i] = np.asarray([[x_pos], [y_pos]]).reshape(-1)
            trial_len_mat[i] = t_trial
        
        #return exp_mat, target_mat, positions
        return exp_mat, target_mat, positions, trial_len_mat