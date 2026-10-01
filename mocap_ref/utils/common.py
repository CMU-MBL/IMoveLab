# name: common.py
# description: Common functions for use


import os, sys
import pickle
import numpy as np 

from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt


# --- Make folder if not existing --- #
def mkfolder(path):

	''' Create a folder in the given path if not exist '''
	
	if not os.path.exists(path):
		os.makedirs(path)


# --- Configurate data processing --- #
def check_filter_config(f_type, dim):

    ''' Check the filter configuration '''

    if f_type in constant_mt.MT_FILTER_LIST[dim.upper()]:
        filter_config_check = True
        error_msg = None
    else:
        filter_config_check = False 
        error_msg = f_type + ' filter does not have ' + dim.upper() + ' option'

    return filter_config_check, error_msg


def get_subject_list(subject, tuning = False):

    ''' Get the list of subjects for processing data '''

    if subject == None:
        if tuning:
            subject_list = constant_common.SUBJECT_LIST_TUNING
        else:
            subject_list = constant_common.SUBJECT_LIST 
    else:
        subject_list = [subject]

    return subject_list


def get_subject_list_long(subject):
    
    ''' Get the list of subjects for processing data (long trials) '''

    if subject == None:
        subject_list = ['4l', '5l', '6l', '13l', '23l'] # HARDCODED: only for 5 subjects
    else:
        subject_list = [subject]

    return subject_list


def get_task_list(task):

	''' Get the list of tasks for processing data '''

	if task == None:
		task_list = list(constant_common.MAPPING_TASK_TO_ID.keys())[1::]
	else:
		task_list = [task]

	return task_list


def get_task_list_long(task):

    ''' Get the list of tasks for processing data (long trials) '''

    if task == None:
        task_list = ['long_walk1', 'long_walk2', 'long_walk3'] # HARDCODED: only for 3 trials
    else:
        task_list = [task]

    return task_list


def get_filter_list(f_type, dim):

    ''' Get the list of filters for processing data '''

    if f_type == None:
        filter_list = constant_mt.MT_FILTER_LIST[dim.upper()]
    else:
        filter_list = [f_type]

    return filter_list


def get_filter_params(f_type, dim):

    ''' Get the tuned parameters of the filter '''

    if f_type.upper() == 'RIANN' or f_type == 'Xsens':
        f_params = None

    else:
        tuning_fn = f'outputs/{f_type.lower()}/{dim.upper()}/tuned_params.pkl'
        with open(tuning_fn, 'rb') as f:
            f_params = pickle.load(f)

    return f_params


def get_filter_params_cf(f_type, dim = '6D'):

    ''' Get the tuned parameters of the filter with constraint feedback '''

    f_params = None

    if f_type.upper() in ['RIANN', 'XSENS']:
        f_params = None

    elif f_type.upper() == 'MAD':
        f_params = [0.04] if dim.upper() == '9D' else [0.4]

    elif f_type.upper() == 'MAH':
        f_params = [0.7, 0.05] if dim.upper() == '9D' else [1, 1.5]

    elif f_type.upper() == 'EKF':
        f_params = [0.02, 0.03, 0.05]

    elif f_type.upper() == 'VQF':
        if dim.upper() == '9D':
            f_params = [2.5, 19]
            
        else:
            f_params = [0.7, 19]

    return f_params


def get_filter_params_for_tuning(f_type):

    ''' Get a mesh of parameters for tuning the filter '''

    if f_type.upper() == 'VQF':
        # tau_a             = np.arange(1, 10, 0.2) # coarse tuning
        # tau_a             = np.arange(0.1, 3, 0.1) # fine tuning

        # combined coarse & fine tuning, but avoid repeated values
        tau_a             = np.unique(np.concatenate((np.arange(1, 10, 0.2), np.arange(0.1, 3, 0.1))))
        tau_m             = np.arange(1, 20, 1)
        filter_params_set = [[param_a, param_m] for param_a in tau_a for param_m in tau_m]

    elif f_type.upper() == 'MAH':
        kp                = np.arange(0.1, 5, 0.2)
        ki                = np.arange(0.05, 2, 0.2)
        filter_params_set = [[kp_val, ki_val] for kp_val in kp for ki_val in ki]

    elif f_type.upper() == 'MAD':
        beta              = np.arange(0.02, 1, 0.02)
        filter_params_set = [[param] for param in beta]

    elif f_type.upper() == 'EKF':
        # sigma_gyro         = np.arange(0.1, 1, 0.1) # coarse tuning
        # sigma_acc          = np.arange(0.1, 1, 0.1)
        # sigma_gyro         = np.arange(0.01, 0.12, 0.01) # fine tuning
        # sigma_acc          = np.arange(0.01, 0.12, 0.01)
        # sigma_gyro         = np.arange(0.001, 0.011, 0.002) # finer tuning
        # sigma_acc          = np.array([0.02, 0.03, 0.04])

        # combined coarse & fine tuning, but avoid repeated values
        sigma_gyro         = np.unique(np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.01, 0.12, 0.01))))
        sigma_acc          = np.unique(np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.01, 0.12, 0.01))))
        sigma_mag          = np.array([0.05, 0.1, 0.5, 0.9])
        filter_params_set  = [[sigma_gyro_val, sigma_acc_val, sigma_mag_val] for sigma_gyro_val in sigma_gyro for sigma_acc_val in sigma_acc for sigma_mag_val in sigma_mag]

    elif f_type.upper() == 'UKF': # NOTE: bad performance, no longer use UKF
        alpha              = np.arange(5e-4, 3e-3, 5e-4)
        beta               = np.arange(1, 3, 0.5)
        kappa              = np.array([-1, 0, 1])
        filter_params_set  = [[alpha_val, beta_val, kappa_val] for alpha_val in alpha for beta_val in beta for kappa_val in kappa]

    elif f_type.upper() == 'RIANN':
        filter_params_set = None

    print(f'Number of filter parameter combinations: {len(filter_params_set) if filter_params_set is not None else 0}')

    return filter_params_set


# From rotation matrices
def rotmat_to_angle(rotmat):

    ''' Convert a rotation matrix to Euler angles '''

    if rotmat.shape == (3, 3): # input is a rotation matrix
        r = R.from_matrix(rotmat) 
    elif rotmat.shape == (4, 4): # input is a homogeneous transformation matrix
        r = R.from_matrix(rotmat[0:3, 0:3])
    else:
        pass 

    sequence = 'ZXY'

    angle 	= r.as_euler(sequence, degrees = True)
    angle_x	= angle[1]
    angle_y	= angle[2]
    angle_z	= angle[0]

    return angle_x, angle_y, angle_z


def rotmat_to_angle_pelvis(rotmat):

    ''' Convert a rotation matrix to Euler angles '''

    if rotmat.shape == (3, 3): # input is a rotation matrix
        r = R.from_matrix(rotmat) 
    elif rotmat.shape == (4, 4): # input is a homogeneous transformation matrix
        r = R.from_matrix(rotmat[0:3, 0:3])
    else:
        pass 

    sequence = 'zxy'

    angle 	= r.as_euler(sequence, degrees = True)
    angle_x	= angle[1]
    angle_y	= angle[2]
    angle_z	= angle[0]

    return angle_x, angle_y, angle_z


def quat_to_angle(quat):

    ''' Convert a quaternion to Euler angles '''

    r = R.from_quat(quat, scalar_first = True)

    sequence = 'ZXY'

    angle 	= r.as_euler(sequence, degrees = True)
    angle_x	= angle[1]
    angle_y	= angle[2]
    angle_z	= angle[0]

    return angle_x, angle_y, angle_z


def low_pass_filter(signal, fs, cutoff = 6, order = 4):

    ''' Low-pass filter a signal '''

    nyquist = 0.5*fs
    normal_cutoff = cutoff/nyquist
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal






