# name: common.py
# description: common utility functions for the whole pipeline


import glob
import os
import pickle

from constants import constant_common
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt


def rot_to_euler(rot_mat):

    ''' convert rotation matrix to euler angles (zxy convention) '''

    sequence = 'ZXY' 

    euler = R.from_matrix(rot_mat).as_euler(sequence, degrees = True)


    return euler


def get_joint_kinematics_from_rot(proximal_segment_to_bone, proximal_lab_to_segment, distal_segment_to_bone, distal_lab_to_segment):

    ''' get joint kinematics from calibrated and oriented body segments '''

    angles_arr = []
    
    num_samples = len(proximal_lab_to_segment)

    for i in range(num_samples):
        proximal_lab_to_bone = proximal_lab_to_segment[i] @ proximal_segment_to_bone
        distal_lab_to_bone   = distal_lab_to_segment[i] @ distal_segment_to_bone

        joint_rot   = np.linalg.inv(proximal_lab_to_bone) @ distal_lab_to_bone
        joint_euler = rot_to_euler(joint_rot[0:3, 0:3])

        angles_arr.append(joint_euler)


    return np.array(angles_arr)


def quat_to_euler(quat, to_deg = True):

    ''' convert quaternion to euler angles '''

    r = R.from_quat(quat, scalar_first = True)

    sequence = 'ZXY'

    euler = r.as_euler(sequence, degrees = to_deg)


    return euler


def get_subject_list(dataset, subject, tuning = False):

    ''' get subject list for processing '''

    if dataset == 'HAKnee':
        if subject == None:
            if tuning:
                subject_list = constant_common.HA_SUBJECT_LIST_TUNING
            else:
                subject_list = constant_common.HA_SUBJECT_LIST_EVAL
        else:
            subject_list = [subject]

    else:
        pass # NOTE: not included in this study


    return subject_list


def get_task_list(dataset, task):

    ''' get task list for processing '''

    if dataset == 'HAKnee':
        if task == None:
            task_list = list(constant_common.HA_TASK_MAPPING.keys())[1::]
        else:
            task_list = [task]

    else:
        pass # NOTE: not included in this study


    return task_list


def get_trial_list(dataset, trial):

    ''' get trial list for processing '''

    if dataset == 'HAKnee':
        if trial == None:
            trial_list = [1, 2, 3]
        else:
            trial_list = [trial]

    else:
        pass # NOTE: not included in this study


    return trial_list


def get_side_list(dataset, side):

    ''' get body side list, i.e., left or right, for processing '''

    if dataset == 'HAKnee':
        if side == None:
            side_list = ['r', 'l']
        else:
            side_list = [side]

    else:
        pass # NOTE: not included in this study


    return side_list


def get_filter_params(dataset, selected_filter):

    ''' get filter parameters for processing '''

    dim = '6d' # only 6d is supported for MC10 IK

    tuning_fn = f'outputs/{dataset}/tuning_out/tuned_params_{selected_filter.lower()}{dim.lower()}.pkl'

    if selected_filter.upper() == 'RIANN':
        filter_params = None
        
    else:
        with open(tuning_fn, 'rb') as f:
            filter_params = pickle.load(f)


    return filter_params


def get_filter_params_for_tuning(selected_filter):

    ''' get filter parameters for tuning '''

    if selected_filter.upper() == 'VQF':
        # tau_a             = np.arange(1, 10, 0.2) # coarse tuning
        # tau_a             = np.arange(0.1, 3, 0.1) # fine tuning

        # combined coarse & fine tuning, but avoid repeated values
        tau_a             = np.unique(np.concatenate((np.arange(1, 10, 0.2), np.arange(0.1, 3, 0.1))))
        filter_params_set = [[param, 1] for param in tau_a]
    elif selected_filter.upper() == 'MAH':
        kp                = np.arange(0.1, 5, 0.2)
        ki                = np.arange(0.05, 2, 0.2)
        filter_params_set = [[kp_val, ki_val] for kp_val in kp for ki_val in ki]
    elif selected_filter.upper() == 'MAD':
        beta              = np.arange(0.02, 1, 0.02)
        filter_params_set = [[param] for param in beta]
    elif selected_filter.upper() == 'EKF':
        # sigma_gyro         = np.arange(0.1, 1, 0.1) # coarse tuning
        # sigma_acc          = np.arange(0.1, 1, 0.1)
        sigma_gyro         = np.arange(0.01, 0.12, 0.01) # fine tuning
        sigma_acc          = np.arange(0.01, 0.12, 0.01)
        # sigma_gyro         = np.arange(0.001, 0.011, 0.002) # finer tuning
        # sigma_acc          = np.array([0.02, 0.03, 0.04])

        # # combined coarse & fine tuning, but avoid repeated values
        # sigma_gyro         = np.unique(np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.01, 0.12, 0.01))))
        # sigma_acc          = np.unique(np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.01, 0.12, 0.01))))
        filter_params_set  = [[sigma_gyro_val, sigma_acc_val, 0.9] for sigma_gyro_val in sigma_gyro for sigma_acc_val in sigma_acc]
    elif selected_filter.upper() == 'UKF':
        alpha              = np.arange(5e-4, 3e-3, 5e-4)
        beta               = np.arange(1, 3, 0.5)
        kappa              = np.array([-1, 0, 1])
        filter_params_set  = [[alpha_val, beta_val, kappa_val] for alpha_val in alpha for beta_val in beta for kappa_val in kappa]
    elif selected_filter.upper() == 'RIANN':
        filter_params_set = None

    print(f'Number of filter parameter combinations: {len(filter_params_set) if filter_params_set is not None else 0}')


    return filter_params_set


def low_pass_filter(data, cutoff = 15, fs = 100, order=4):

    ''' apply low-pass Butterworth filter to the data '''
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)


    return y


def format_filter_params(filter_params):

    ''' Format filter params to avoid floating point precision issues '''


    return '_'.join([f'{param:g}' for param in filter_params])


def find_existing_folder(base_path, filter_params):

    '''Find existing folder matching filter params, handling precision issues.'''
    
    filter_params_str = format_filter_params(filter_params)
    exact_path = os.path.join(base_path, f'p_{filter_params_str}')
    if os.path.exists(exact_path):
        return f'p_{filter_params_str}'
    
    pattern = os.path.join(base_path, f'p_{"_".join(["*" for _ in filter_params])}')
    matching_folders = glob.glob(pattern)
    if matching_folders:
        return os.path.basename(matching_folders[0])
    
    
    return f'p_{filter_params_str}' 


