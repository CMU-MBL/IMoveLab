# name: preprocessing_mt.py
# description: preprocess data from MTw Awinda (Xsens)


import pandas as pd 
import numpy as np 
from tqdm import tqdm 

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt


# --- Obtain all data from Xsens sensors --- #
# Get data path of a single sensor
def get_data_path_mt(subject, task, sensor_id):

    ''' Get data path (experiments 2 and 3) '''

    mt_fn = f'{constant_common.IN_LAB_PATH}s{subject}/{constant_common.IN_MT_PATH}{constant_common.LAB_TASK_NAME_MAP[task]}-000_{sensor_id}{constant_common.MT_EXTENSION}'

    return mt_fn


# Load data from a single sensor
def load_data_mt(mt_fn):
    
    ''' Load data from a single sensor (i.e., a .txt file) '''

    with open(mt_fn, 'r') as f:
        txt    = f.readlines()
        data_flag = -1
        data_id   = 0
        while data_flag == -1:
            data_flag = txt[data_id].find('\t')
            data_id += 1
        header = txt[data_id - 1].split('\t')
        
    temp_data = np.genfromtxt(mt_fn, delimiter = '\t', skip_header = data_id)
    temp_data = np.delete(temp_data, 1, 1)

    header.pop(1)
    header[-1] = header[-1][0:-1]

    mt_data = pd.DataFrame(temp_data, columns = header)

    return mt_data


# Get data from all sensors
def get_all_data_mt(subject, task, sensor_config):

    ''' Get data from all sensors (experiments 2 and 3) '''

    data_mt = {}

    for sensor_name in tqdm(sensor_config.keys()):
        sensor_id            = constant_mt.LAB_IMU_NAME_MAP[sensor_config[sensor_name]]
        mt_fn                = get_data_path_mt(subject, task, sensor_id)
        data_mt[sensor_name] = load_data_mt(mt_fn)

    return data_mt


# --- Synchronize data from all sensors --- #
# Update the package counter since it reset after overflow (at 65535)
def update_packet_counter_mt(data_mt):

    ''' Update the package counter of all sensors '''

    for sensor_name in data_mt.keys():
        offset = 0
        unwrap_counter = [data_mt[sensor_name].loc[0, 'PacketCounter']]

        for i in range(1, data_mt[sensor_name].shape[0]):
            if data_mt[sensor_name].loc[i, 'PacketCounter'] < data_mt[sensor_name].loc[i - 1, 'PacketCounter']:
                offset += 65536
            unwrap_counter.append(data_mt[sensor_name].loc[i, 'PacketCounter'] + offset)

        data_mt[sensor_name]['PacketCounter'] = unwrap_counter

    return data_mt


# To fix the frame drops
def get_data_length_mt(data_mt):

    ''' Obtain data length from IMU sensors '''

    for sensor_name in data_mt.keys():
        start_counter = data_mt[sensor_name]['PacketCounter'].to_numpy()[0]
        stop_counter  = data_mt[sensor_name]['PacketCounter'].to_numpy()[-1]

    return start_counter, stop_counter


def match_data_mt(data_mt):

    ''' Synchronize data from all sensors ''' 

    data_mt = update_packet_counter_mt(data_mt)

    start_counter, stop_counter = get_data_length_mt(data_mt)
    nan_id                      = np.arange(start_counter, stop_counter, 1)
    nan_id                      = np.array(nan_id)
    nan_arr                     = np.nan*np.ones(nan_id.shape[0])
    nan_frame                   = pd.DataFrame({'temp': nan_arr}, index = nan_id)
    
    matched_data_mt = {}
    for sensor_name in data_mt.keys():
        temp_frame = data_mt[sensor_name].set_index(data_mt[sensor_name]['PacketCounter']).iloc[:, 1::]
        temp_frame = temp_frame.join(nan_frame, how = 'outer')
        temp_frame = temp_frame.interpolate(method = 'linear', limit_area = 'inside')

        interp_frame = temp_frame.loc[nan_id, :]
        interp_frame = interp_frame.iloc[1:-1, 0:-1]

        matched_data_mt[sensor_name]         = 1*interp_frame.reset_index()
        matched_data_mt[sensor_name].columns = data_mt[sensor_name].columns
        matched_data_mt[sensor_name]         = matched_data_mt[sensor_name].interpolate(method = 'linear')

    return matched_data_mt


