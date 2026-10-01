# name: run_mt.py
# description: run unconstrained IK on MTw data


import numpy as np 
import quaternion
import pickle

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt, constant_mocap
from utils import common
from utils.mt import preprocessing_mt, calibration_mt, ik_mt



def mt_ik(selected_setup, f_type, dim, subject, task, source = 'mt', tuning = False):

    ''' Get joint angles from MTw data '''
    
    if source == 'mt':
        subject_list = common.get_subject_list(subject, tuning)
        task_list    = common.get_task_list(task)
        
    elif source == 'mt_long':
        subject_list = common.get_subject_list_long(subject)
        task_list    = common.get_task_list_long(task)

    filter_list  = common.get_filter_list(f_type, dim.upper())

    for f_type in filter_list:
        print('*** Filter ' + f_type)

        if not tuning:
            filter_params_set = [common.get_filter_params(f_type, dim)]
        else:
            filter_params_set = common.get_filter_params_for_tuning(f_type)

        print('=' *50)
        print(f'Running MC10 IK with filter {f_type} and dim {dim} ...')
        if tuning:
            print('Tuning mode: ON')
        else:
            print('Tuning mode: OFF')
        print('=' *50)
        print()

        for subject in subject_list:
            print('*** Subject ' + str(subject))
            print('*** Sensor axes ' + dim.upper())

            for f_params in filter_params_set:
                print('*** Filter parameters: ' + str(f_params))

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

                if selected_setup[0].upper() == 'F':
                    walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Y'].to_numpy())
                else:
                    walking_period = calibration_mt.get_walking_4_calib(data_walking_mt['shank_r']['Gyr_Z'].to_numpy())
                
                jumping_period = [0, data_jumping_mt['pelvis']['Gyr_Y'].shape[0]]

                seg2sens = calibration_mt.sensor_to_segment_mt(data_static_mt, data_walking_mt, walking_period, data_jumping_mt, jumping_period, selected_setup)

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


                        print('- Estimate joint angles')
                        if source == 'mt':
                            orientation_mt, time_mt = ik_mt.get_imu_orientation_mt(data_main_mt, f_type = f_type, fs = constant_mt.MT_SAMPLING_RATE, dim = dim.upper(), params = f_params, get_time = True)
                        elif source == 'mt_long':
                            orientation_mt, time_mt = ik_mt.get_imu_orientation_mt(data_main_mt, f_type = f_type, fs = constant_mocap.MOCAP_SAMPLING_RATE, dim = dim.upper(), params = f_params, get_time = True)


                        if dim.upper() == '6D':
                            if source == 'mt':
                                orientation_mt = calibration_mt.correct_random_6D_orientation(initial_orientation, orientation_mt)
                            elif source == 'mt_long':
                                orientation_mt = calibration_mt.correct_random_6D_orientation(initial_orientation, orientation_mt, fs = constant_mocap.MOCAP_SAMPLING_RATE)
                        
                        ja_mt = ik_mt.get_all_ja_mt(seg2sens, orientation_mt)

                        if source == 'mt_long':
                            pass 
                        else:
                            print('- Apply synchronization') 
                            sync_fn = constant_common.OUT_SYNC_INFO + 'sync_info_s' + str(subject) + '_' + selected_task + '.pkl'
                            with open(sync_fn, 'rb') as f:
                                sync_info = pickle.load(f)

                            if sync_info['first_start'] == 'imu':
                                for joint in ja_mt.keys():
                                    ja_mt[joint] = ja_mt[joint][sync_info['shifting_id']:]


                        print('- Save results of ' + selected_task)
                        if tuning:
                            filter_params_str = '_'.join([str(round(param, 3)) for param in f_params])
                            output_path = f'outputs/{f_type.lower()}/{dim.upper()}/tuning/{filter_params_str}/{subject}/'

                            common.mkfolder(output_path)
                            if selected_setup == 'mm':
                                filename = f'{output_path}{selected_task}.pkl'
                            else:
                                filename = f'{output_path}{selected_task}_{selected_setup}.pkl'
                            with open(filename, 'wb') as f:
                                pickle.dump(ja_mt, f)

                        else:
                            if source == 'mt_long':
                                output_path = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/ik/{subject}/'
                                
                                common.mkfolder(output_path)
                                filename = f'{output_path}{selected_task}.pkl'
                                with open(filename, 'wb') as f:
                                    pickle.dump(ja_mt, f)
                            
                            else:
                                output_path = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik/{subject}/'

                                common.mkfolder(output_path)
                                filename = f'{output_path}{selected_task}.pkl'
                                with open(filename, 'wb') as f:
                                    pickle.dump(ja_mt, f)
                                
                                if selected_setup == 'mm':
                                    output_path_time = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/run_time/{subject}/'

                                    common.mkfolder(output_path_time)
                                    filename = f'{output_path_time}{selected_task}.pkl'
                                    with open(filename, 'wb') as f:
                                        pickle.dump(time_mt, f)

                        print('\n\n\n\n\n')

                    except:
                        print('*** Error in processing ' + selected_task)





