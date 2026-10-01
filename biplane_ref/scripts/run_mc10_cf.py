# name: run_mc10_cf.py
# description: Obtain IMU kinematics with constraint feedback (CF) for the MC10 dataset 


import quaternion
import os 
import pickle 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from datetime import datetime, timezone
from easydict import EasyDict
from vqf import PyVQF
from ahrs.filters import Madgwick, Mahony, EKF
from ahrs.common.orientation import acc2q
from scipy.spatial.transform import Rotation as R

from utils.mc10 import mc10_processing, mc10_ik, mc10_calibration
from utils.mocap import fp_processing
from utils import common
from utils import coupling_models
from constants import constant_mocap, constant_mc10, constant_meta, constant_common



def get_data_chunk(data, duration):

    ''' Get a chunk of data based on the specified duration (start and stop time) '''

    data_chunk = {}
    
    for sensor_name in data.keys():
        data_chunk[sensor_name] = data[sensor_name].iloc[duration[0]:duration[1]]
    

    return data_chunk


def low_pass_filter(signal, fs, cutoff = 6, order = 4):

    nyquist = 0.5*fs
    normal_cutoff = cutoff/nyquist
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    filtered_signal = filtfilt(b, a, signal)


    return filtered_signal


def get_sensor_transform(initial_orientation, Q, num_static_samples):
    ''' to align sensors with body segments based on the perfect standing assumption for 6D filters '''

    static_orientation = 1*quaternion.as_quat_array(Q[0:num_static_samples]) 


    return initial_orientation * np.mean(static_orientation).conjugate()


def get_joint_rotation(segment_prox, segment_dist, seg2sens_prox, seg2sens_dist):
    ''' to get the joint rotation (quaternion) from the two segments '''
    
    segment_prox_aligned = segment_prox * quaternion.from_rotation_matrix(seg2sens_prox).conjugate()
    segment_dist_aligned = segment_dist * quaternion.from_rotation_matrix(seg2sens_dist).conjugate()

    joint_quat = segment_prox_aligned.conjugate() * segment_dist_aligned


    return joint_quat


def get_all_joints(seg2sens, sensor_frame, timestep = None):
    ''' to get joint rotations of all joints at a given timestep'''

    joint_frame = {}

    joint_frame['knee_r']  = get_joint_rotation(sensor_frame['thigh_r'][timestep], sensor_frame['shank_r'][timestep], seg2sens['thigh_r'], seg2sens['shank_r'])
    joint_frame['knee_l']  = get_joint_rotation(sensor_frame['thigh_l'][timestep], sensor_frame['shank_l'][timestep], seg2sens['thigh_l'], seg2sens['shank_l'])


    return joint_frame


def correct_nonsagittal_knee(joint_quat, seg2sens, joint_aligned, sensor_transforms, timestep, joint, prox, dist, alpha, coupling = None):
    ''' correct knee adduction/abduction and internal/external rotation based on the knee coupling (Walker, Rovick & Robertson, J Biomech 1988;21:965-974, eqs 1-2; fitted to cadaver data of Reuben et al. 1986)'''


    joint_rot = R.from_quat(quaternion.as_float_array(joint_quat[joint]), scalar_first = True)
    knee_flex, knee_add, knee_rot = joint_rot.as_euler('ZXY', degrees = True)

    knee_flex *= -1
    if joint[-1] == 'l': 
        knee_add *= -1 
        knee_rot *= -1

    if coupling is None: # original Walker et al. (1988) coupling
        coupling = coupling_models.get_coupling('walker')
    knee_add_coupled = coupling[0](knee_flex)
    knee_rot_coupled = coupling[1](knee_flex)

    knee_add_error = alpha * (knee_add_coupled - knee_add)
    knee_rot_error = alpha * (knee_rot_coupled - knee_rot)

    if joint[-1] == 'l':
        knee_add_error *= -1 
        knee_rot_error *= -1

    rot_knee_error = R.from_euler('ZXY', np.array([0, knee_add_error, knee_rot_error]), degrees = True)
    q_knee_error = rot_knee_error.as_quat(scalar_first = True)
    q_knee_error /= np.linalg.norm(q_knee_error) 
    q_knee_error = quaternion.as_quat_array(q_knee_error)

    corrected_joint         = joint_quat[joint] * q_knee_error
    corrected_joint_aligned = joint_aligned[prox][timestep] * quaternion.from_rotation_matrix(seg2sens[prox]).conjugate() * corrected_joint * quaternion.from_rotation_matrix(seg2sens[dist])
    corrected_joint_raw     = quaternion.as_float_array(sensor_transforms[dist].conjugate() * corrected_joint_aligned)
    corrected_joint_raw     /= np.linalg.norm(corrected_joint_raw)


    return corrected_joint_aligned, corrected_joint_raw


def init_orientation(data_main_mt, num_samples):
    ''' to get the initial orientation for state-estimation filters based on accelerometer data at the first timestep (perfect standing assumption) '''

    orientation = {}

    for sensor_name in data_main_mt.keys():
        orientation[sensor_name] = np.zeros((num_samples, 4))

        acc0 = data_main_mt[sensor_name].loc[0, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()
        orientation[sensor_name][0] = acc2q(acc0) 


    return orientation


def one_step_update(filter, data, Q, sensor_name, t, enable_vqf = False):
    ''' NOTE: only for filters from the AHRS library '''

    acc_t = data[sensor_name].loc[t, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()
    gyr_t = data[sensor_name].loc[t, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()

    try:
        Q_ = filter.updateIMU(Q, gyr = gyr_t, acc = acc_t)
    except:
        Q_ = filter.update(Q, gyr = gyr_t, acc = acc_t) # in case of EKF


    return Q_


def mc10_ik_cf_main(dataset, subject, task, trial, side, filter_type, dim, tuning = False, filter_params = None, knee_gain = 0.9, savefig = False, coupling = 'walker'):

    ''' Run IK for the MC10 Biostamp data '''

    alpha = 1*knee_gain
    suffix = coupling_models.folder_suffix(coupling)
    print(f'Knee coupling: {coupling}')

    filter_type     = filter_type.upper()
    selected_filter = filter_type

    if not tuning:
        filter_params_set = [common.get_filter_params(dataset, filter_type)]
    else:
        filter_params_set = common.get_filter_params_for_tuning(filter_type)

    print('=' *50)
    print(f'Running MC10 IK with filter {selected_filter} and dim {dim} ...')
    if tuning:
        print('Tuning mode: ON')
    else:
        print('Tuning mode: OFF')
    print('=' *50)
    print()

    subject_list = common.get_subject_list(dataset, subject, tuning)
    task_list    = common.get_task_list(dataset, task)
    trial_list   = common.get_trial_list(dataset, trial)
    side_list    = common.get_side_list(dataset, side)

    for filter_params in filter_params_set:

        print(f'Filter parameters: {filter_params}\n')

        for subject in subject_list:

            print(f'*** Subject {subject}')
            coupling_fns = coupling_models.get_coupling(coupling, subject)
            test = 1    

            print('Getting MC10 data ...')
            mc10_data = mc10_processing.get_mc10_data(dataset, subject)


            for task in task_list:

                for trial in trial_list:

                    for side in side_list:

                        try:

                            selected_task = EasyDict(side = side, trial = trial, task = task)


                            walking_duration = constant_meta.WALKING_DURATION[str(subject)]

                            running_duration = [constant_meta.TASK_DURATION[str(subject)][selected_task.task][selected_task.side][f't{selected_task.trial}'][0],
                                                constant_meta.TASK_DURATION[str(subject)][selected_task.task][selected_task.side][f't{selected_task.trial}'][1]]

                            static_duration = [constant_meta.STATIC_DURATION[str(subject)][constant_meta.STATIC_SIDE[str(subject)]][0],
                                               constant_meta.STATIC_DURATION[str(subject)][constant_meta.STATIC_SIDE[str(subject)]][1]]

                    
                            static_data  = get_data_chunk(mc10_data, static_duration)
                            walking_data = get_data_chunk(mc10_data, walking_duration)
                            running_data = get_data_chunk(mc10_data, running_duration)

                            running_data = {sensor_name: pd.concat([static_data[sensor_name], running_data[sensor_name]], ignore_index=True) for sensor_name in mc10_data.keys()}

                            walking_period = mc10_calibration.get_walking_4_calib(walking_data['shank_r']['Gyr_Z'].to_numpy())
                            seg2sens = mc10_calibration.sensor_to_segment_mc10(static_data, walking_data, walking_period)

                            initial_orientation = {}
                            for sensor_name in seg2sens.keys():
                                initial_orientation[sensor_name] = quaternion.from_rotation_matrix(np.identity(3))*quaternion.from_rotation_matrix(seg2sens[sensor_name])

                            num_samples        = running_data['thigh_r'].shape[0]
                            num_static_samples = static_data['thigh_r'].shape[0]

                            print(f'Number of samples in the static data: {num_static_samples}')
                            print(f'Number of samples in the running data: {num_samples}')

                            if selected_filter == 'VQF':

                                filter = {}
                                for sensor_name in mc10_data.keys():
                                    filter[sensor_name] = PyVQF(gyrTs = 1.0/constant_mc10.PROCESSING_RATE, tauAcc = filter_params[0], tauMag = filter_params[1])

                                sensor_transforms = {}

                                filter_raw = {}
                                for sensor_name in mc10_data.keys():
                                    filter_raw[sensor_name] = np.empty((num_samples, 4))

                                filter_aligned    = {}

                                for sensor_name in running_data.keys():
                                    for i in range(10):
                                        for phantom_timestep in range(num_static_samples):
                                            gyr_t = running_data[sensor_name].loc[phantom_timestep, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
                                            acc_t = running_data[sensor_name].loc[phantom_timestep, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()

                                            filter[sensor_name].update(gyr = gyr_t, acc = acc_t)
                                            filter_raw[sensor_name][phantom_timestep] = 1*filter[sensor_name].getQuat6D()

                                timestep = 0

                                for sensor_name in filter_raw.keys(): 
                                    filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])

                                # apply the sensor transform to the whole trial
                                for sensor_name in running_data.keys():
                                    sensor_transforms[sensor_name] = get_sensor_transform(initial_orientation[sensor_name], filter_raw[sensor_name], num_static_samples)

                                    filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])
                                    for k in range(num_samples):
                                        filter_aligned[sensor_name][k] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][k]


                                last_check = 1*timestep
                                update_interval = 1

                                while timestep < num_samples:

                                    if timestep % 1000 == 0:
                                        print(f'current timestep: {timestep} / {num_samples}')

                                    # update the filter_raw and filter_aligned every timestep
                                    for sensor_name in running_data.keys():
                                        gyr_t = running_data[sensor_name].loc[timestep, ['Gyr_X','Gyr_Y','Gyr_Z']].to_numpy()
                                        acc_t = running_data[sensor_name].loc[timestep, ['Acc_X','Acc_Y','Acc_Z']].to_numpy()

                                        filter[sensor_name].update(gyr = gyr_t, acc = acc_t)
                                        filter_raw[sensor_name][timestep] = 1*filter[sensor_name].getQuat6D()

                                        filter_aligned[sensor_name][timestep] = 1*quaternion.as_quat_array(filter_raw[sensor_name][timestep])
                                        filter_aligned[sensor_name][timestep] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][timestep]

                                    if timestep - last_check >= update_interval:

                                        joint_quat = get_all_joints(seg2sens, filter_aligned, timestep = timestep)

                                        corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_r', prox = 'thigh_r', dist = 'shank_r', alpha = alpha, coupling = coupling_fns)
                                        filter_aligned['shank_r'][timestep] = 1*corrected_joint_aligned
                                        filter_raw['shank_r'][timestep]     = 1*corrected_joint_raw
                                        state_r = filter['shank_r'].state
                                        g_r                     = state_r['gyrQuat']
                                        state_r['accQuat']      = filter[sensor_name].quatMultiply(corrected_joint_raw/np.linalg.norm(corrected_joint_raw), filter[sensor_name].quatConj(g_r)) # NOTE: reset the internal state of the filter too
                                        filter['shank_r'].state = state_r

                                        corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_l', prox = 'thigh_l', dist = 'shank_l', alpha = alpha, coupling = coupling_fns)
                                        filter_aligned['shank_l'][timestep] = 1*corrected_joint_aligned
                                        filter_raw['shank_l'][timestep]     = 1*corrected_joint_raw
                                        state_l = filter['shank_l'].state
                                        g_l                     = state_l['gyrQuat']
                                        state_l['accQuat']      = filter[sensor_name].quatMultiply(corrected_joint_raw/np.linalg.norm(corrected_joint_raw), filter[sensor_name].quatConj(g_l)) # NOTE: reset the internal state of the filter too
                                        filter['shank_l'].state = state_l

                                        last_check = 1*timestep

                                    timestep += 1

                                print('Getting MC10 knee kinematics ...')
                                knee_kinematics = mc10_ik.get_knee_kinematics_mc10(seg2sens, filter_aligned)

                                for joint_name in knee_kinematics.keys():
                                    knee_kinematics[joint_name] = low_pass_filter(knee_kinematics[joint_name], constant_mc10.PROCESSING_RATE, cutoff = 6, order = 4)

                                output_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/')

                                with open(output_fn, 'wb') as f:
                                    pickle.dump(knee_kinematics, f)

                            else:

                                if selected_filter == 'MAD':
                                    filter = Madgwick(frequency = constant_mc10.PROCESSING_RATE, gain = filter_params[0])
                                elif selected_filter == 'MAH':
                                    filter = Mahony(frequency = constant_mc10.PROCESSING_RATE) # default params
                                elif selected_filter == 'EKF':
                                    filter = EKF(frequency = constant_mc10.PROCESSING_RATE) # default params

                                sensor_transforms = {}
                                filter_raw = init_orientation(running_data, num_samples)
                                filter_aligned    = {}

                                timestep = 1

                                # get sensor transform
                                while timestep < num_static_samples:
                                    for sensor_name in running_data.keys():
                                        filter_raw[sensor_name][timestep] = one_step_update(filter, running_data, filter_raw[sensor_name][timestep-1], sensor_name, timestep)
                                    
                                    timestep += 1

                                # apply the sensor transform to the whole trial
                                for sensor_name in running_data.keys():
                                    sensor_transforms[sensor_name] = get_sensor_transform(initial_orientation[sensor_name], filter_raw[sensor_name], num_static_samples)

                                    filter_aligned[sensor_name] = 1*quaternion.as_quat_array(filter_raw[sensor_name])
                                    for k in range(num_samples):
                                        filter_aligned[sensor_name][k] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][k]

                                last_check = 1*timestep
                                update_interval = 1

                                while timestep < num_samples:

                                    if timestep % 1000 == 0:
                                        print(f'current timestep: {timestep} / {num_samples}')

                                    # update the filter_raw and filter_aligned every timestep
                                    for sensor_name in running_data.keys():
                                        filter_raw[sensor_name][timestep]     = one_step_update(filter, running_data, filter_raw[sensor_name][timestep-1], sensor_name, timestep)
                                        filter_aligned[sensor_name][timestep] = 1*quaternion.as_quat_array(filter_raw[sensor_name][timestep])
                                        filter_aligned[sensor_name][timestep] = sensor_transforms[sensor_name] * filter_aligned[sensor_name][timestep]

                                    if timestep - last_check >= update_interval:

                                        joint_quat = get_all_joints(seg2sens, filter_aligned, timestep = timestep)

                                        corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_r', prox = 'thigh_r', dist = 'shank_r', alpha = alpha, coupling = coupling_fns)
                                        filter_aligned['shank_r'][timestep] = 1*corrected_joint_aligned
                                        filter_raw['shank_r'][timestep]     = 1*corrected_joint_raw

                                        corrected_joint_aligned, corrected_joint_raw = correct_nonsagittal_knee(joint_quat, seg2sens, filter_aligned, sensor_transforms, timestep, joint = 'knee_l', prox = 'thigh_l', dist = 'shank_l', alpha = alpha, coupling = coupling_fns)
                                        filter_aligned['shank_l'][timestep] = 1*corrected_joint_aligned
                                        filter_raw['shank_l'][timestep]     = 1*corrected_joint_raw

                                        last_check = 1*timestep


                                    timestep += 1

                                print('Getting MC10 knee kinematics ...')
                                knee_kinematics = mc10_ik.get_knee_kinematics_mc10(seg2sens, filter_aligned)

                                for joint_name in knee_kinematics.keys():
                                    knee_kinematics[joint_name] = low_pass_filter(knee_kinematics[joint_name], constant_mc10.PROCESSING_RATE, cutoff = 6, order = 4)

                                output_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}_constrained_{int(alpha*100)}p{suffix}/ik/{subject}/mc10/')

                                with open(output_fn, 'wb') as f:
                                    pickle.dump(knee_kinematics, f)


                        except:

                            print(f'No MC10 data for subject {subject}, task {task}, trial {trial}')












