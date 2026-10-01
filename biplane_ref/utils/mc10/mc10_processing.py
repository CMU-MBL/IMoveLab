# name: mc10_processing.py
# description: processing functions for the MC10 Biostamp data


import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import constant_mc10, constant_common


def format_data_columns(data):

    ''' format the data columns '''

    data_columns = list(data.columns)
    data_columns = [column.split(' ')[0] for column in data_columns]
    data_columns[1] += '_X'
    data_columns[2] += '_Y'
    data_columns[3] += '_Z'

    data.columns = data_columns


    return data


def get_mc10_path(dataset, subject, sensor_name):

    ''' get the path to the mc10 data '''

    path = os.path.join(constant_common.DATA_PATH, constant_common.IMU_PATH, dataset, str(subject).zfill(2), constant_mc10.SENSOR_NAME_MAP[sensor_name])


    return path


def interpolate_frame_drop(data, timestamp):

    ''' resample the gyro and accel data '''

    nan_arr = np.nan*np.ones(timestamp.shape[0])
    nan_frame = pd.DataFrame({'temp': nan_arr,}, index = timestamp)

    interpolated_data = data.set_index(data['Timestamp']).iloc[:, 1::]
    interpolated_data = interpolated_data.join(nan_frame, how = 'outer')
    interpolated_data = interpolated_data.interpolate(method = 'linear', limit_area = 'inside')
    interpolated_data = interpolated_data.iloc[:, 0:-1]
    interpolated_data = interpolated_data.reset_index()


    return interpolated_data


def load_data(path):

    ''' load the data from the path '''

    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    fn_accel        = os.path.join(path, 'accel.csv')
    data_accel      = pd.read_csv(fn_accel)
    data_accel      = format_data_columns(data_accel)
    accel_timestamp = data_accel['Timestamp'].to_numpy()

    fn_gyro         = os.path.join(path, 'gyro.csv')
    data_gyro       = pd.read_csv(fn_gyro)
    data_gyro       = format_data_columns(data_gyro)
    gyro_timestamp  = data_gyro['Timestamp'].to_numpy()

    timestamp  = np.union1d(accel_timestamp, gyro_timestamp)
    data_accel = interpolate_frame_drop(data_accel, timestamp)
    data_gyro  = interpolate_frame_drop(data_gyro, timestamp)

    acc = data_accel[['Accel_X', 'Accel_Y', 'Accel_Z']].to_numpy()
    acc = constant_mc10.GRAV_ACC * np.transpose(np.dot(Rz, acc.T))
    
    gyr = data_gyro[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].to_numpy()
    gyr = np.deg2rad(np.transpose(np.dot(Rz, gyr.T)))

    timestamp = timestamp.reshape([timestamp.shape[0], 1])

    data_one_sensor = pd.DataFrame(np.concatenate([timestamp, acc, gyr], axis = 1), columns = constant_mc10.IMU_DATA_HEADERS)


    return data_one_sensor
    

def resample_data(data, ft, start_time, end_time):

    ''' resample the data '''

    ts = 1*constant_mc10.TO_MICRO / ft

    nan_id    = np.arange(start_time, end_time, ts)
    nan_arr   = np.nan*np.ones(nan_id.shape[0])
    nan_frame = pd.DataFrame({'temp': nan_arr,}, index = nan_id)
    
    resampled_data = {}
    for sensor_name in constant_mc10.SENSOR_NAME_MAP.keys():
        temp_frame = data[sensor_name].set_index(data[sensor_name]['Timestamp']).iloc[:, 1::]
        temp_frame = temp_frame.join(nan_frame, how = 'outer')
        temp_frame = temp_frame.interpolate(method = 'linear', limit_area = 'inside')

        interp_frame = temp_frame.loc[nan_id, :]
        interp_frame = interp_frame.iloc[1:-1, 0:-1]

        resampled_data[sensor_name] = 1*interp_frame.reset_index()

        resampled_data[sensor_name].columns = constant_mc10.IMU_DATA_HEADERS


    return resampled_data


def sync_data(data):

    ''' sync the data '''

    late_start_time = 0
    early_stop_time = 0

    count = 0
    for sensor_name in data.keys():
        if count == 0:
            late_start_time = data[sensor_name]['Timestamp'].to_numpy()[0]
            early_stop_time = data[sensor_name]['Timestamp'].to_numpy()[-1]

        else:
            if late_start_time < data[sensor_name]['Timestamp'].to_numpy()[0]:
                late_start_time = data[sensor_name]['Timestamp'].to_numpy()[0]

            if early_stop_time > data[sensor_name]['Timestamp'].to_numpy()[-1]:
                early_stop_time = data[sensor_name]['Timestamp'].to_numpy()[-1]

        count += 1

    synced_data = {}
    for sensor_name in data.keys():
        start_id = np.where(data[sensor_name]['Timestamp'].to_numpy() >= late_start_time)
        stop_id  = np.where(data[sensor_name]['Timestamp'].to_numpy() <= early_stop_time)
        duration_id = np.intersect1d(start_id, stop_id)

        synced_data[sensor_name] = data[sensor_name].iloc[duration_id, :]
        synced_data[sensor_name] = synced_data[sensor_name].reset_index(drop = True)


    return synced_data, late_start_time, early_stop_time


def get_mc10_data(dataset, subject):

    ''' get the mc10 data '''

    mc10_data = {}

    for sensor_name in tqdm(constant_mc10.SENSOR_NAME_MAP.keys()):
        data_path = get_mc10_path(dataset, subject, sensor_name)
        temp_data = load_data(data_path)
        mc10_data[sensor_name] = 1*temp_data

    mc10_data, late_start_time, early_stop_time = sync_data(mc10_data)

    mc10_data = resample_data(mc10_data, constant_mc10.PROCESSING_RATE, late_start_time, early_stop_time)


    return mc10_data















