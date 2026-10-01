# name: mc10_ik.py 
# description: supporting functions for IK of the MC10 Biostamp data


import numpy as np
import quaternion 

from tqdm import tqdm

import sys, os

from utils import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import constant_mc10, constant_common
from utils.mc10 import sfa



def get_mc10_orientation(mc10_data, f_type, fs = constant_mc10.PROCESSING_RATE, params = None):

    ''' get the sensor orientation (quaternion) from the IMU data '''

    print(f'Applying {f_type} filter to get sensor orientation (quaternion), fs = {fs} Hz ...')
    mc10_orientation = {}

    for sensor_name in tqdm(mc10_data.keys()):
        gyr = mc10_data[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
        acc = mc10_data[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        
        if f_type == 'VQF':
            temp_estimate = sfa.apply_vqf(gyr, acc, fs, params)
        elif f_type == 'MAH':
            temp_estimate = sfa.apply_mahony(gyr, acc, fs, params)
        elif f_type == 'MAD':
            temp_estimate = sfa.apply_madgwick(gyr, acc, fs, params)
        elif f_type == 'EKF':
            temp_estimate = sfa.apply_ekf(gyr, acc, fs, params)
        elif f_type == 'UKF':
            temp_estimate = sfa.apply_ukf(gyr, acc, fs, params)
        elif f_type == 'RIANN':
            temp_estimate = sfa.apply_riann(gyr, acc, fs)

        if f_type == 'VQF':
            mc10_orientation[sensor_name] = quaternion.as_quat_array(temp_estimate['quat6D'])
        elif f_type == 'RIANN':
            mc10_orientation[sensor_name] = quaternion.as_quat_array(temp_estimate)
        else:
            mc10_orientation[sensor_name] = quaternion.as_quat_array(temp_estimate.Q)
            
            
    return mc10_orientation


def average_orientation(orientation_data):

    ''' average the sensor orientation (quaternion), typically over the static period '''

    orientation_avg = {}
    
    for sensor_name in orientation_data.keys():
        orientation_avg[sensor_name] = orientation_data[sensor_name].mean(axis=0)
    

    return orientation_avg


def get_seg2sens_6D(static_orientation):

    ''' get the segment-to-sensor calibration for 6D filters based on the perfect standing assumption '''

    init_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    init_orientation = quaternion.from_rotation_matrix(init_rot)

    seg2sens = {}
    for sensor_name in static_orientation.keys():
        seg2sens[sensor_name] = init_orientation.conjugate() * static_orientation[sensor_name]


    return seg2sens


def get_angles(o1, o2, s2s1, s2s2, calibration_flag = True):

    ''' get the joint angles from the sensor orientations and segment-to-sensor calibrations '''
    
    N = o1.shape[0]

    angle_arr = []

    s2s1 = quaternion.from_rotation_matrix(s2s1)
    s2s2 = quaternion.from_rotation_matrix(s2s2)

    if calibration_flag:
        s1 = [o1[i] * s2s1.conjugate() for i in range(N)]
        s2 = [o2[i] * s2s2.conjugate() for i in range(N)]

    else:
        s1 = 1 * o1
        s2 = 1 * o2

    rot_arr = [s1[i].conjugate() * s2[i] for i in range(N)]
    rot_arr = quaternion.as_float_array(rot_arr)
    angle_arr = [common.quat_to_euler(rot) for rot in rot_arr]
    angle_arr = np.array(angle_arr)

    assert angle_arr.shape == (N, 3), 'Incorrect data shape'


    return angle_arr


def get_knee_kinematics_mc10(seg2sens, mc10_orientation):

    ''' get the knee kinematics from the sensor orientations and segment-to-sensor calibrations '''
    
    knee_kinematics = {}

    temp_knee_r = get_angles(mc10_orientation['thigh_r'], mc10_orientation['shank_r'], seg2sens['thigh_r'], seg2sens['shank_r'])
    knee_kinematics['knee_flexion_r']   = constant_common.IK_SIGN['knee_flexion_r'] * temp_knee_r[:, 0]
    knee_kinematics['knee_adduction_r'] = constant_common.IK_SIGN['knee_adduction_r'] * temp_knee_r[:, 1]
    knee_kinematics['knee_rotation_r']  = constant_common.IK_SIGN['knee_rotation_r'] * temp_knee_r[:, 2]

    temp_knee_l = get_angles(mc10_orientation['thigh_l'], mc10_orientation['shank_l'], seg2sens['thigh_l'], seg2sens['shank_l'])
    knee_kinematics['knee_flexion_l']   = constant_common.IK_SIGN['knee_flexion_l'] * temp_knee_l[:, 0]
    knee_kinematics['knee_adduction_l'] = constant_common.IK_SIGN['knee_adduction_l'] * temp_knee_l[:, 1]
    knee_kinematics['knee_rotation_l']  = constant_common.IK_SIGN['knee_rotation_l'] * temp_knee_l[:, 2]


    return knee_kinematics

















