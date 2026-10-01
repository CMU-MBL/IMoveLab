# name: preprocessing_mocap.py
# description: preprocessing functions for marker-based motion capture data


import numpy as np 
import pandas as pd
from scipy import signal 

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common


# --- Obtain all data from marker-based motion capture --- #
# Get data path of the mocap file
def get_data_path_mocap(subject, task):

    ''' Get data path (experiments 2 and 3) '''

    mocap_fn = f'{constant_common.IN_LAB_PATH}s{subject}/{constant_common.IN_MOCAP_PATH}{constant_common.LAB_TASK_NAME_MAP[task]}{constant_common.MOCAP_EXTENSION}'

    return mocap_fn


# Get mocap data
def get_data_mocap(subject, task):

    ''' Obtain the mocap data '''
    
    mocap_fn = get_data_path_mocap(subject, task)

    mocap_data = pd.read_csv(mocap_fn, skiprows = 3, low_memory = False)
    mocap_data = mocap_data.iloc[:, 1:]
    mocap_data = mocap_data.iloc[2:, :]

    names_pos = list(mocap_data.columns)
    names_pos = [name.split(':')[1][0:4] for name in names_pos[1:]]
    names_pos = [''] + names_pos
    names_ax  = mocap_data.iloc[0, :]
    names     = []

    for i in range(len(names_pos)):
        names.append(names_pos[i] + ' ' + names_ax[i])
    names[0] = 'Time'

    mocap_data = mocap_data.iloc[1:, :]
    mocap_data.columns = names

    mocap_data = mocap_data.reset_index()
    mocap_data = mocap_data.iloc[:, 1:]
    mocap_data = mocap_data.astype('float64')

    return mocap_data


# --- Low pass filter mocap data --- #
def lowpass_filter_mocap(mocap_data, fs, fc, fo):

    ''' Low-pass filter the mocap data '''

    Wn = fc*2/fs
    b, a = signal.butter(fo, Wn, btype = 'low')
    f_mocap_data = signal.filtfilt(b, a, mocap_data, axis = 0)
    f_mocap_data = pd.DataFrame(f_mocap_data, columns = mocap_data.columns)
    f_mocap_data.iloc[:, 0] = mocap_data.iloc[:, 0]

    return f_mocap_data


# --- Resample mocap data --- #
def resample_mocap(mocap_data, ft):

    ''' Resample mocap data to match IMU data (e.g., experiment 2) '''

    ts = 1/ft
    nan_id = np.arange(mocap_data['Time'].to_numpy()[0], mocap_data['Time'].to_numpy()[-1], ts)
    nan_arr = np.nan*np.ones(nan_id.shape[0])
    nan_frame = pd.DataFrame({'temp': nan_arr}, index = nan_id)

    temp_frame = mocap_data.set_index(mocap_data['Time']).iloc[:, 1::]
    temp_frame = temp_frame.join(nan_frame, how = 'outer')
    temp_frame = temp_frame.interpolate(method = 'cubic', limit_area = 'inside')
    interp_frame = temp_frame.loc[nan_id, :]
    interp_frame = interp_frame.iloc[1:-1, 0:-1]

    resampled_mocap_data = 1*interp_frame.reset_index()
    resampled_mocap_data.columns = mocap_data.columns 

    return resampled_mocap_data


# --- Get average mocap/IMU data during static --- #
def get_avg_data(static_dt):

    ''' Average mocap/IMU data '''

    temp_dt = 1*static_dt.mean(axis = 0)
    avg_dt  = pd.DataFrame(temp_dt.transpose().values.reshape(1, -1), columns = temp_dt.index)

    return avg_dt

