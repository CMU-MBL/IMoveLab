# name: calibration_mt.py
# description: sensor-to-body alignment for IMUs


import numpy as np 

from numpy.linalg import norm 
from sklearn.decomposition import PCA 
from tqdm import tqdm 

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_mt, constant_common
from utils.events import event_mt


# --- Get PCA axis --- #
def get_pc1_ax_mt(data):

    ''' Get the rotation axis during walking (for thighs/shanks/feet) or squat (for pelvis) using PCA '''

    data = data - np.mean(data, axis = 0)
    pca  = PCA(n_components = 3)
    pca.fit(data)

    pc1_ax = 1*pca.components_[0]

    return pc1_ax


# --- Sensor-to-segment alignment (calibration) --- #
# Get walking period for calibration
def get_walking_4_calib(shank_walking_gyr_r):

    ''' Get walking period for calibration '''

    gait_events = event_mt.detect_gait_events(shank_walking_gyr_r)
    
    period = [gait_events['ms_index'][10], gait_events['ms_index'][18]]

    return period


# Calibration (when using squat or jumping for pelvis and sternum)
def sensor_to_segment_mt(data_static, data_walking, walking_period, data_squat, squat_period, selected_setup):

    ''' Obtain transformation from segment-to-sensor '''

    seg2sens = {}

    for sensor_name in tqdm(data_static.keys()):
        static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
        vy         = np.mean(static_acc, axis = 0)
        fy         = vy/norm(vy)

        side = sensor_name[-1]
        if sensor_name == 'chest':
            fx = np.ones(3) 
            fy = np.ones(3) 
            fz = np.ones(3) # ignore as we do not use 
            
        elif sensor_name == 'pelvis':
            squat_gyr = 1*data_squat[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            squat_gyr = squat_gyr[squat_period[0]:squat_period[1], :]
            pc1_ax    = get_pc1_ax_mt(squat_gyr)

            if pc1_ax[1] > 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)

        elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if pc1_ax[1] < 0:
                pc1_ax = (-1)*pc1_ax
            
            vx = np.cross(fy, pc1_ax)
            fx = vx/norm(vx)

            vz = np.cross(fx, fy)
            fz = vz/norm(vz)
        
        else:
            walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
            walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
            pc1_ax      = get_pc1_ax_mt(walking_gyr)

            if 'thigh' in sensor_name:
                dir = selected_setup[1]
            elif 'shank' in sensor_name:
                dir = selected_setup[0]
            else:
                pass

            if dir.upper() == 'F':
                if pc1_ax[1] < 0:
                    pc1_ax = (-1)*pc1_ax
                
                vx = np.cross(fy, pc1_ax)
                fx = vx/norm(vx)

                vz = np.cross(fx, fy)
                fz = vz/norm(vz)

            else:
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

# # Calibration (when using toe-touching for pelvis and sternum instead of squat or jumping)
# def sensor_to_segment_mt(data_static, data_walking, walking_period, data_toetouch):
#     ''' Obtain transformation from segment-to-sensor

#     Args:
#         + data_static (dict of pd.DataFrame): static data for the vertical axis
#         + data_walking (dict of pd.DataFrame): walking data for thigh/shank/foot rotational axis
#         + data_squat (dict of pd.DataFrame): squat data for pelvis rotational axis

#     Returns:
#         + seg2sens (dict of pd.DataFrame): segment-to-sensor transformation
#     '''
#     seg2sens = {}

#     for sensor_name in tqdm(data_static.keys()):
#         static_acc = 1*data_static[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
#         vy         = np.mean(static_acc, axis = 0)
#         fy         = vy/norm(vy)

#         side = sensor_name[-1]
#         if sensor_name == 'stern':            
#             # print('+ stern sensor is calibrated')
#             toetouch_acc = 1*data_toetouch[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
#             temp_z       = np.mean(toetouch_acc, axis = 0)
#             temp_z       = temp_z/norm(temp_z)

#             vz = np.cross(fy, temp_z)
#             fz = vz/norm(vz)

#             vx = np.cross(fy, fz)
#             fx = vx/norm(vx)
            
#         elif sensor_name == 'pelvis':
#             toetouch_acc = 1*data_toetouch[sensor_name][['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
#             temp_z       = np.mean(toetouch_acc, axis = 0)
#             temp_z       = temp_z/norm(temp_z)

#             vz = np.cross(fy, temp_z)
#             fz = vz/norm(vz)

#             vx = np.cross(fy, fz)
#             fx = vx/norm(vx)

#         elif (sensor_name == 'foot_r') or (sensor_name == 'foot_l'):
#             walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
#             walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
#             pc1_ax      = get_pc1_ax_mt(walking_gyr)

#             if pc1_ax[1] < 0:
#                 pc1_ax = (-1)*pc1_ax
            
#             vx = np.cross(fy, pc1_ax)
#             fx = vx/norm(vx)

#             vz = np.cross(fx, fy)
#             fz = vz/norm(vz)
        
#         else:
#             walking_gyr = 1*data_walking[sensor_name][['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
#             walking_gyr = walking_gyr[walking_period[0]:walking_period[1], :]
#             pc1_ax      = get_pc1_ax_mt(walking_gyr)

#             if pc1_ax[-1] < 0:
#                 pc1_ax = (-1)*pc1_ax
            
#             if side == 'r':
#                 vx = np.cross(fy, pc1_ax)
#             else:
#                 vx = np.cross(pc1_ax, fy)
            
#             fx = vx/norm(vx)

#             vz = np.cross(fx, fy)
#             fz = vz/norm(vz)
        
#         seg2sens[sensor_name] = np.array([fx, fy, fz])

#     return seg2sens


# Correct random 6D orientation
def correct_random_6D_orientation(initial_orientation, main_orientation_mt, fs = constant_mt.MT_SAMPLING_RATE):

    ''' Correct random 6D orientation '''

    sensor_transform = {}
    main_orientation_mt_corrected = {}

    num_static_samples = constant_common.STATIC_STANDING_PERIOD*fs

    for sensor_name in main_orientation_mt.keys():
        sensor_transform[sensor_name] = initial_orientation[sensor_name] * np.mean(main_orientation_mt[sensor_name][0:num_static_samples]).conjugate()

        main_orientation_mt_corrected[sensor_name] = []
        for i in range(len(main_orientation_mt[sensor_name])):
            main_orientation_mt_corrected[sensor_name].append(sensor_transform[sensor_name] * main_orientation_mt[sensor_name][i])

        main_orientation_mt_corrected[sensor_name] = np.array(main_orientation_mt_corrected[sensor_name])

    return main_orientation_mt_corrected



