# name: calibration_mc10.py
# description: calibration for MC10 sensors


import numpy as np 

from numpy.linalg import norm
from sklearn.decomposition import PCA 
from tqdm import tqdm 

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import constant_mc10, constant_common
from utils.mc10 import mc10_event


# Get walking period for calibration
def get_walking_4_calib(shank_walking_gyr_r):
    ''' Get walking period for calibration '''

    gait_events = mc10_event.detect_gait_events(shank_walking_gyr_r)
    
    period = [gait_events['ms_index'][10], gait_events['ms_index'][18]]

    return period


# --- Get PCA axis --- #
def get_pc1_ax_mc10(data):

    ''' Get the rotation axis during walking (for thighs/shanks/feet) or squat (for pelvis) using PCA '''
    
    data = data - np.mean(data, axis = 0)
    pca  = PCA(n_components = 3)
    pca.fit(data)

    pc1_ax = 1*pca.components_[0]

    return pc1_ax


# --- Sensor-to-segment alignment (calibration) --- #
def sensor_to_segment_mc10(data_static, data_walking, walking_period):

    ''' Obtain transformation from segment-to-sensor '''

    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        
        walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
        walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
        pc1_ax      = get_pc1_ax_mc10(walking_gyr)

        if pc1_ax[-1] < 0:
            pc1_ax = (-1)*pc1_ax
        
        if side == 'r':
            vx = np.cross(fy, pc1_ax)
        else:
            vx = np.cross(pc1_ax, fy)
        
        fx = vx/norm(vx)

        vz = np.cross(fx, fy)
        fz = vz/norm(vz)
        
        seg2sens[sensor_name] = np.array([fx, fy, fz])

    return seg2sens


# Correct random 6D orientation
def correct_random_6D_orientation(initial_orientation, main_orientation_mc10, fs = constant_mc10.PROCESSING_RATE):

    ''' Correct random 6D orientation '''

    sensor_transform = {}
    main_orientation_mc10_corrected = {}

    num_static_samples = int(constant_common.STATIC_STANDING_PERIOD*fs)

    for sensor_name in main_orientation_mc10.keys():
        sensor_transform[sensor_name] = initial_orientation[sensor_name] * np.mean(main_orientation_mc10[sensor_name][0:num_static_samples]).conjugate()

        main_orientation_mc10_corrected[sensor_name] = []
        for i in range(len(main_orientation_mc10[sensor_name])):
            main_orientation_mc10_corrected[sensor_name].append(sensor_transform[sensor_name] * main_orientation_mc10[sensor_name][i])

        main_orientation_mc10_corrected[sensor_name] = np.array(main_orientation_mc10_corrected[sensor_name])

    return main_orientation_mc10_corrected


def get_calib_imu_data(imu_data, seg2sens):

    ''' Apply calibration to raw IMU data '''
    
    cal_imu_data = {}

    for sensor_name in tqdm(imu_data.keys()):
        cal_imu_data[sensor_name] = 1*imu_data[sensor_name]
        cal_imu_data[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']] = np.dot(seg2sens[sensor_name], imu_data[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].T).T
        cal_imu_data[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']] = np.dot(seg2sens[sensor_name], imu_data[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].T).T

    return cal_imu_data






























