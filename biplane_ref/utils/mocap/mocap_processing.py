# name: mocap_processing.py
# description: processing functions for the mocap data


import ezc3d

import pandas as pd
import numpy as np
import scipy.signal as signal

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import constant_common, constant_mocap


def get_mocap_path(dataset, subject, test, task):

    ''' get path to the mocap data '''

    if dataset == 'HAKnee':
        subject_str = 'Subject' + str(subject).zfill(2) if subject < 10 else 'Subject' + str(subject)
        task_str    = task.side.upper() + constant_common.HA_TASK_MAPPING[task.task] + str(task.trial)

    elif dataset == 'Navio':
        subject_str = 'Subject' + str(subject)
        if 'static' in task.task or 'walk' in task.task:
            task_str = task.side.upper() + constant_common.NAVIO_TASK_MAPPING[task.task] + str(task.trial)
        else:
            task_str = task.side.lower() + constant_common.NAVIO_TASK_MAPPING[task.task] + str(task.trial) 

    test_str    = 'Test' + str(test)

    path = os.path.join(constant_common.DATA_PATH, constant_common.MOCAP_PATH, dataset, subject_str, test_str, task_str + constant_common.MOCAP_EXTENSION)


    return path


def get_mocap_data(dataset, subject, test, task):

    ''' get mocap data from c3d file '''

    path = get_mocap_path(dataset, subject, test, task); print(path); print()

    c3d_data = ezc3d.c3d(path)

    marker_traj = {}

    marker_list = c3d_data['parameters']['POINT']['LABELS']['value']
    marker_list = [marker for marker in marker_list if "*" not in marker]
    
    # remove 'Subject:' if exists in marker_list
    for i in range(len(marker_list)):
        if 'Subject' in marker_list[i]:
            marker_list[i] = marker_list[i].replace(f'Subject{subject}:', '').strip()

    # get sampling rate
    fs = c3d_data['parameters']['POINT']['RATE']['value'][0]
    print(f'Mocap sampling rate: {fs} Hz')

    for marker in marker_list:
        marker_traj[marker] = c3d_data['data']['points'][:, marker_list.index(marker), :].T/1e3

        marker_traj[marker] = pd.DataFrame(
            marker_traj[marker], 
            columns=[f"{marker} X", f"{marker} Y", f"{marker} Z", "1"]
        )
        marker_traj[marker].drop(columns = ['1'], inplace = True) # drop the constant column

    mocap_data = pd.concat(marker_traj.values(), axis = 1)

    num_frames = mocap_data.shape[0]
    time       = np.arange(0, num_frames/fs, 1/fs)
    time       = pd.DataFrame(time, columns = ['Time'])

    mocap_data = pd.concat([time, mocap_data], axis = 1)


    return mocap_data, marker_list
    

def mocap_lowpass_filter(mocap_data, fs, fc, fo = 4):

    ''' lowpass filter the mocap data '''
    
    Wn   = fc*2/fs
    b, a = signal.butter(fo, Wn, btype = 'low')

    f_mocap_data            = signal.filtfilt(b, a, mocap_data, axis = 0)
    f_mocap_data            = pd.DataFrame(f_mocap_data, columns = mocap_data.columns)
    f_mocap_data.iloc[:, 0] = 1*mocap_data.iloc[:, 0]
    

    return f_mocap_data


def mocap_resample(mocap_data, ft):

    ''' resample the mocap data '''

    ts        = 1/ft
    nan_id    = np.arange(mocap_data['Time'].to_numpy()[0], mocap_data['Time'].to_numpy()[-1], ts)
    nan_arr   = np.nan*np.ones(nan_id.shape[0])
    nan_frame = pd.DataFrame({'temp': nan_arr}, index = nan_id)

    temp_frame = mocap_data.set_index(mocap_data['Time']).iloc[:, 1::]
    temp_frame = temp_frame.join(nan_frame, how = 'outer')
    temp_frame = temp_frame.interpolate(method = 'linear', limit_area = 'inside')

    interp_frame = temp_frame.loc[nan_id, :]
    interp_frame = interp_frame.iloc[:, 0:-1]

    r_mocap_data         = 1*interp_frame.reset_index()
    r_mocap_data.columns = mocap_data.columns


    return r_mocap_data


def get_average_mocap(mocap_data):

    ''' get average mocap data '''

    temp_dt = 1*mocap_data.mean(axis = 0, skipna = True)
    avg_mocap_data = pd.DataFrame(temp_dt.transpose().values.reshape(1, -1), columns = temp_dt.index)


    return avg_mocap_data


def get_mocap_masking(mocap_data, dataset):

    ''' get the mask of the valid mocap data '''

    if dataset == 'HAKnee':
        considered_marker_list = constant_mocap.HAKNEE_THIGH_CLUSTER_IK_LIST_R + constant_mocap.HAKNEE_SHANK_CLUSTER_IK_LIST_R + constant_mocap.HAKNEE_THIGH_CLUSTER_IK_LIST_L + constant_mocap.HAKNEE_SHANK_CLUSTER_IK_LIST_L

        mocap_masking = np.ones(mocap_data.shape[0])

        for marker in considered_marker_list:
            mocap_masking = mocap_masking & (~np.isnan(mocap_data[marker + ' X']))
            mocap_masking = mocap_masking & (~np.isnan(mocap_data[marker + ' Y']))
            mocap_masking = mocap_masking & (~np.isnan(mocap_data[marker + ' Z']))

        mocap_masking = mocap_masking.astype(bool)

        mocap_masking[mocap_masking == False] = np.nan

    elif dataset == 'Navio':
        pass # NOTE: not included in this study


    return mocap_masking





