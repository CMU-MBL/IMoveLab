# name: synchronization.py
# description: Synchronize mocap and IMU data


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_mt
from utils.eval.metrics import get_rmse


# --- Get vertical acceleration from a specific IMU --- #
def get_vertical_acc_mt(one_imu_data, source = 'mt'):

    ''' Get vertical acceleration of an IMU data '''

    if source == 'mt':
        vertical_acc_mt = 1*one_imu_data['Acc_X'].to_numpy()
    elif source == 'mvn':
        vertical_acc_mt = 1*one_imu_data[:, 0]

    vertical_acc_mt -= constant_mt.EARTH_G_ACC

    return vertical_acc_mt


# --- Get vertical acceleration from a spcific marker --- #
def get_vertical_acc_mocap(one_marker_data, fs = constant_mt.MT_SAMPLING_RATE):
    
    ''' Get vertical acceleration of a marker '''

    vertical_acc_mocap = 1*one_marker_data.to_numpy()
    vertical_acc_mocap = np.diff(vertical_acc_mocap)/(1.0/fs)
    vertical_acc_mocap = np.diff(vertical_acc_mocap)/(1.0/fs)

    return vertical_acc_mocap


# --- Identify the hop period with mocap data --- #
def get_hop_id_mocap(one_mocap_data):

    ''' Get hop id from mocap '''    

    possible_id, _ = find_peaks(one_mocap_data, height = 5)

    hop_id_mocap = 0
    peak_count   = 0
    while hop_id_mocap < 200:
        hop_id_mocap = 1*possible_id[peak_count]
        peak_count  += 1

    return hop_id_mocap


# --- Get information for sync'ing --- #
def get_sync_info(mocap_data, pelvis_mt_data, window = 120, iters = 1500, fs = constant_mt.MT_SAMPLING_RATE, source = 'mt'):

    ''' Get information for sync'ing IMU and mocap data '''

    shifting_id = 0
    prev_err    = 999

    pelvis_vertical_acc_mt    = get_vertical_acc_mt(pelvis_mt_data, source)
    pelvis_vertical_acc_mocap = get_vertical_acc_mocap(mocap_data['RPS2 Y'], fs)
    hop_id_mocap              = get_hop_id_mocap(pelvis_vertical_acc_mocap[0:int(len(pelvis_vertical_acc_mocap)/2)])


    error = []
    for i in range(iters):
        start_mocap = hop_id_mocap - int(window/2)
        stop_mocap = hop_id_mocap + int(window/2)
        if i > start_mocap:
            break
        start_imu = start_mocap - i
        stop_imu = stop_mocap - i
        curr_err = get_rmse(pelvis_vertical_acc_mocap[start_mocap:stop_mocap], pelvis_vertical_acc_mt[start_imu:stop_imu])
        error.append(curr_err)

        if curr_err < prev_err:
            shifting_id = i + 2
            prev_err = curr_err

    mocap_error = min(error)
    mocap_shifting_id = shifting_id

    error = []
    for i in range(iters):
        start_mocap = hop_id_mocap - int(window/2)
        stop_mocap = hop_id_mocap + int(window/2)
        start_imu = start_mocap + i
        stop_imu = stop_mocap + i
        curr_err = get_rmse(pelvis_vertical_acc_mocap[start_mocap:stop_mocap], pelvis_vertical_acc_mt[start_imu:stop_imu])
        error.append(curr_err)

        if curr_err < prev_err:
            if i >=2:
                shifting_id = i - 2
            else:
                shifting_id = 0
            prev_err = curr_err

    mt_err = min(error)
    mt_shifting_id = shifting_id


    if mocap_error < mt_err:
        shifting_id = mocap_shifting_id
        first_start = 'mocap'
    else:
        shifting_id = mt_shifting_id
        first_start = 'imu'

    return first_start, shifting_id






