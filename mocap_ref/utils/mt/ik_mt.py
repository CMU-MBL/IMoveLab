# name: ik_mt.py
# description: perform IK to obtain joint kinematics from IMU data


import numpy as np
import quaternion
import time

from tqdm import tqdm

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt
from utils.mt import sfa, sfa_cf
from utils import common


# Get orientation from all sensors
def get_imu_orientation_mt(imu_data_mt, f_type, initial_orientation = None, seg2sens = None, fs = constant_mt.MT_SAMPLING_RATE, dim = '9D', params = None, get_time = False, cf_flag = False):

    ''' Get orientation from all sensors '''

    imu_orientation_mt = {}
    time_mt            = {}
    print('fs = %s' %(fs))

    if cf_flag:
        if f_type == 'VQF':
            imu_orientation_mt, time_mt = sfa_cf.apply_vqf_cf(imu_data_mt, seg2sens, initial_orientation, dim, fs, params)
        elif f_type in ['MAD', 'MAH', 'EKF']:
            imu_orientation_mt, time_mt = sfa_cf.apply_ahrs_cf(imu_data_mt, f_type, seg2sens, initial_orientation, dim, fs, params)
        else:
            raise ValueError('Constraint feedback is only implemented for VQF, MAD, MAH, and EKF')

    else:
        for sensor_name in tqdm(imu_data_mt.keys()):
            start_time = time.time()

            if f_type == 'Xsens':
                imu_orientation_mt[sensor_name] = quaternion.as_quat_array(imu_data_mt[sensor_name][['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']].to_numpy())
            else:
                gyr = imu_data_mt[sensor_name][['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
                acc = imu_data_mt[sensor_name][['Acc_X','Acc_Y','Acc_Z']].to_numpy()
                if dim == '9D':
                    mag = imu_data_mt[sensor_name][['Mag_X','Mag_Y','Mag_Z']].to_numpy()
                else:
                    mag = None

                if f_type == 'VQF':
                    temp_estimation = sfa.apply_vqf(gyr, acc, mag, dim, fs, params)
                elif f_type == 'MAD':
                    temp_estimation = sfa.apply_madgwick(gyr, acc, mag, dim, fs, params)
                elif f_type == 'MAH':
                    temp_estimation = sfa.apply_mahony(gyr, acc, mag, dim, fs, params)
                elif f_type == 'EKF':
                    temp_estimation = sfa.apply_ekf(gyr, acc, mag, dim, fs, params)
                elif f_type == 'RIANN':
                    temp_estimation = sfa.apply_riann(gyr, acc, fs)

                if f_type == 'VQF':
                    imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation['quat' + dim])
                elif f_type == 'RIANN':
                    imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation)
                else:
                    imu_orientation_mt[sensor_name] = quaternion.as_quat_array(temp_estimation.Q)

            time_mt[sensor_name] = (time.time() - start_time)/imu_orientation_mt[sensor_name].shape[0]

    if get_time:
        return imu_orientation_mt, time_mt
         
    else:
        return imu_orientation_mt


# --- Get joint angles between two adjacent segments --- #
def get_ja(sframe_1, sframe_2, s2s_1, s2s_2, c_flag = True):

    ''' Get joint angles from the provided orientation '''

    N = sframe_1.shape[0]
    imu_ja = []

    s2s_1 = quaternion.from_rotation_matrix(s2s_1)
    s2s_2 = quaternion.from_rotation_matrix(s2s_2)

    if c_flag:
        segment_1 = [sframe_1[i]*s2s_1.conjugate() for i in range(N)]
        segment_2 = [sframe_2[i]*s2s_2.conjugate() for i in range(N)]
    else:
        segment_1 = 1*sframe_1
        segment_2 = 1*sframe_2
    
    joint_rot = [segment_1[i].conjugate()*segment_2[i] for i in range(N)]
    joint_rot = quaternion.as_float_array(joint_rot)
    imu_ja    = [common.quat_to_angle(joint) for joint in joint_rot]
    imu_ja    = np.array(imu_ja)
    
    assert imu_ja.shape == (N, 3), 'Incorrect data shape'

    return imu_ja


# --- Get 5-DOF lower limb joint angles --- #
def get_all_ja_mt(seg2sens, orientation_mt, c_flag = True):

    ''' Obtain all joint angles from IMUs '''

    mt_ja = {}
    
    temp_hip_l   = get_ja(orientation_mt['pelvis'], orientation_mt['thigh_l'], seg2sens['pelvis'], seg2sens['thigh_l'], c_flag = c_flag)
    temp_knee_l  = get_ja(orientation_mt['thigh_l'], orientation_mt['shank_l'], seg2sens['thigh_l'], seg2sens['shank_l'], c_flag = c_flag)
    temp_ankle_l = get_ja(orientation_mt['shank_l'], orientation_mt['foot_l'], seg2sens['shank_l'], seg2sens['foot_l'], c_flag = c_flag)

    mt_ja['hip_adduction_l']   = constant_common.JA_SIGN['hip_adduction_l']*temp_hip_l[:, 0]
    mt_ja['hip_rotation_l']    = constant_common.JA_SIGN['hip_rotation_l']*temp_hip_l[:, 1]
    mt_ja['hip_flexion_l']     = constant_common.JA_SIGN['hip_flexion_l']*temp_hip_l[:, 2]
    mt_ja['knee_adduction_l']  = constant_common.JA_SIGN['knee_adduction_l']*temp_knee_l[:, 0]
    mt_ja['knee_rotation_l']   = constant_common.JA_SIGN['knee_rotation_l']*temp_knee_l[:, 1]
    mt_ja['knee_flexion_l']    = constant_common.JA_SIGN['knee_flexion_l']*temp_knee_l[:, 2]
    mt_ja['ankle_adduction_l'] = constant_common.JA_SIGN['ankle_adduction_l']*temp_ankle_l[:, 0]
    mt_ja['ankle_rotation_l']  = constant_common.JA_SIGN['ankle_rotation_l']*temp_ankle_l[:, 1]
    mt_ja['ankle_flexion_l']   = constant_common.JA_SIGN['ankle_flexion_l']*temp_ankle_l[:, 2]

    temp_hip_r   = get_ja(orientation_mt['pelvis'], orientation_mt['thigh_r'], seg2sens['pelvis'], seg2sens['thigh_r'], c_flag = c_flag)
    temp_knee_r  = get_ja(orientation_mt['thigh_r'], orientation_mt['shank_r'], seg2sens['thigh_r'], seg2sens['shank_r'], c_flag = c_flag)
    temp_ankle_r = get_ja(orientation_mt['shank_r'], orientation_mt['foot_r'], seg2sens['shank_r'], seg2sens['foot_r'], c_flag = c_flag)

    mt_ja['hip_adduction_r']   = constant_common.JA_SIGN['hip_adduction_r']*temp_hip_r[:, 0]
    mt_ja['hip_rotation_r']    = constant_common.JA_SIGN['hip_rotation_r']*temp_hip_r[:, 1]
    mt_ja['hip_flexion_r']     = constant_common.JA_SIGN['hip_flexion_r']*temp_hip_r[:, 2]
    mt_ja['knee_adduction_r']  = constant_common.JA_SIGN['knee_adduction_r']*temp_knee_r[:, 0]
    mt_ja['knee_rotation_r']   = constant_common.JA_SIGN['knee_rotation_r']*temp_knee_r[:, 1]
    mt_ja['knee_flexion_r']    = constant_common.JA_SIGN['knee_flexion_r']*temp_knee_r[:, 2]
    mt_ja['ankle_adduction_r'] = constant_common.JA_SIGN['ankle_adduction_r']*temp_ankle_r[:, 0]
    mt_ja['ankle_rotation_r']  = constant_common.JA_SIGN['ankle_rotation_r']*temp_ankle_r[:, 1]
    mt_ja['ankle_flexion_r']   = constant_common.JA_SIGN['ankle_flexion_r']*temp_ankle_r[:, 2]

    return mt_ja




