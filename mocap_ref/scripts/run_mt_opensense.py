# name: run_mt_opensense.py
# description: run unconstrained IK on MTw data constrained by OpenSense biomechanical model


import numpy as np 
import quaternion
import pickle
import time

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt, constant_mocap
from utils import common
from utils.mt import preprocessing_mt, calibration_mt, ik_mt, ik_os


def mt_ik_opensense(selected_setup, f_type, dim, subject, task, source = 'mt'):

    ''' Get joint angles from MTw data constrained by OpenSense biomechanical model '''

    if source == 'mt':
        subject_list = common.get_subject_list(subject)
        task_list    = common.get_task_list(task)
        filter_list  = common.get_filter_list(f_type, dim.upper())
    elif source == 'mt_long':
        subject_list = common.get_subject_list_long(subject)
        task_list    = common.get_task_list_long(task)
        filter_list  = common.get_filter_list(f_type, dim.upper())

    for f_type in filter_list:
        print('*** Filter ' + f_type)

        for subject in subject_list:
            print('*** Subject ' + str(subject))
            print('*** Sensor axes ' + dim.upper())

            f_params = common.get_filter_params(f_type, dim)

            sensor_config  = {'pelvis': 'PELVIS', 
                              'foot_r': 'FOOT_R', 'shank_r': 'SHANK_R_' + selected_setup[0].upper(), 'thigh_r': 'THIGH_R_' + selected_setup[1].upper(),
                              'foot_l': 'FOOT_L', 'shank_l': 'SHANK_L_' + selected_setup[0].upper(), 'thigh_l': 'THIGH_L_' + selected_setup[1].upper()}

            print('- Find sensor-to-segment calibration')
            task_static     = 'static'
            data_static_mt  = preprocessing_mt.get_all_data_mt(subject, task_static, sensor_config)
            data_static_mt  = preprocessing_mt.match_data_mt(data_static_mt) 
            task_walking    = 'treadmill_walking' 
            data_walking_mt = preprocessing_mt.get_all_data_mt(subject, task_walking, sensor_config)
            task_jumping    = 'cmj' 
            data_jumping_mt = preprocessing_mt.get_all_data_mt(subject, task_jumping, sensor_config)

            if selected_setup == 'front':
                walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Y'].to_numpy())
            else:
                walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Z'].to_numpy())
            
            jumping_period = [0, data_jumping_mt['pelvis']['Gyr_Y'].shape[0]]

            seg2sens = calibration_mt.sensor_to_segment_mt(data_static_mt, data_walking_mt, walking_period, data_jumping_mt, jumping_period, selected_setup)

            os_model = 'Rajagopal_2015'
            print('- Apply the customized sensor-to-segment calibration to the OpenSim model: ' + os_model)
            ik_os.os_calibration_customized(seg2sens, os_model)

            if dim.upper() == '6D':
                print('(Perfect standing assumption for 6D filters)')
                initial_orientation = {}
                for sensor_name in seg2sens.keys():
                    initial_orientation[sensor_name] = quaternion.from_rotation_matrix(np.identity(3))*quaternion.from_rotation_matrix(seg2sens[sensor_name])


            for selected_task in task_list:
                print('*** TASK: ' + selected_task)

                try: 
                    data_main_mt = preprocessing_mt.get_all_data_mt(subject, selected_task, sensor_config)
                    data_main_mt = preprocessing_mt.match_data_mt(data_main_mt)

                    if source == 'mt':
                        opensense_catch = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/opensense_catch/{subject}/'
                    elif source == 'mt_long':
                        opensense_catch = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/opensense_catch/{subject}/'
                    common.mkfolder(opensense_catch)

                    print('- Run IK for the calibrated model')
                    if source == 'mt':
                        main_orientation_mt = ik_mt.get_imu_orientation_mt(data_main_mt, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim.upper(), params = f_params)
                    elif source == 'mt_long':
                        main_orientation_mt = ik_mt.get_imu_orientation_mt(data_main_mt, f_type = f_type, fs = constant_mocap.MOCAP_SAMPLING_RATE, dim = dim.upper(), params = f_params)

                    if dim.upper() == '6D':
                        if source == 'mt':
                            main_orientation_mt = calibration_mt.correct_random_6D_orientation(initial_orientation, main_orientation_mt)
                        elif source == 'mt_long':
                            main_orientation_mt = calibration_mt.correct_random_6D_orientation(initial_orientation, main_orientation_mt, fs = constant_mocap.MOCAP_SAMPLING_RATE)
                    
                    if source == 'mt':
                        ik_os.convert_imu_orientation_to_os(selected_task, main_orientation_mt, fs = constant_mt.MT_SAMPLING_RATE, stat_flag = False, catch_path = opensense_catch)
                    elif source == 'mt_long':
                        ik_os.convert_imu_orientation_to_os(selected_task, main_orientation_mt, fs = constant_mocap.MOCAP_SAMPLING_RATE, stat_flag = False, catch_path = opensense_catch)

                    orientation_fn = f'{selected_task}_orientation.sto'
                    time_mt = {}
                    start_time = time.time()
                    ik_os.os_ik(orientation_fn, os_model, False, catch_path = opensense_catch) 
                    time_mt['ik'] = time.time() - start_time

                    ik_fn     = f'ik_{selected_task}_orientation.mot'
                    imu_os_ja = ik_os.get_all_ja_os(ik_fn, os_model, catch_path = opensense_catch) 


                    if source == 'mt_long':
                        pass
                    else:
                        print('- Apply synchronization')
                        sync_fn = f'{constant_common.OUT_SYNC_INFO}sync_info_s{subject}_{selected_task}.pkl'
                        with open(sync_fn, 'rb') as f:
                            sync_info = pickle.load(f)

                        if sync_info['first_start'] == 'imu':
                            for joint in imu_os_ja.keys():
                                imu_os_ja[joint] = imu_os_ja[joint][sync_info['shifting_id']:]

                    print('- Save results of ' + selected_task)
                    if source == 'mt':
                        output_path = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik_opensense/{subject}/'
                    elif source == 'mt_long':
                        output_path = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/ik_opensense/{subject}/'
                        
                    common.mkfolder(output_path)
                    filename = f'{output_path}{selected_task}.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(imu_os_ja, f)

                    if source == 'mt':
                        if selected_setup == 'mm':
                            output_path_time = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/run_time_opensense/{subject}/'

                            common.mkfolder(output_path_time)
                            filename = f'{output_path_time}{selected_task}.pkl'
                            with open(filename, 'wb') as f:
                                pickle.dump(time_mt, f) 
                    
                    elif source == 'mt_long':
                        if selected_setup == 'mm':
                            output_path_time = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/run_time_opensense/{subject}/'
                            
                            common.mkfolder(output_path_time)
                            filename = f'{output_path_time}{selected_task}.pkl'
                            with open(filename, 'wb') as f:
                                pickle.dump(time_mt, f) 

                except:
                    print('*** Error in processing ' + selected_task)






