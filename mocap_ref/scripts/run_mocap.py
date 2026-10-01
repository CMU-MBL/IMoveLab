# name: run_mocap.py
# description: run unconstrained IK on mocap data


import pickle

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mt, constant_mocap
from utils import common
from utils.mocap import preprocessing_mocap, ik_mocap



def mocap_ik(subject, task, source = 'mt', tuning = False):

    ''' Obtain IK from marker-based data '''

    if source == 'mt':
        subject_list = common.get_subject_list(subject, tuning = tuning)
        task_list    = common.get_task_list(task)
    elif source == 'mt_long':
        subject_list = common.get_subject_list_long(subject)
        task_list    = common.get_task_list_long(task)

    for subject in subject_list:
        print('*** Subject ' + str(subject))

        print('- Static calibration for the mocap data')
        task_static           = 'static'
        data_static_mocap     = preprocessing_mocap.get_data_mocap(subject, task_static)
        data_static_mocap     = data_static_mocap.interpolate(method = 'cubic')
        data_static_mocap     = data_static_mocap.fillna(value = constant_common.VERY_HIGH_NUMBER)
        data_static_mocap_avg = preprocessing_mocap.get_avg_data(data_static_mocap)

        print('- Calibrate mocap')
        static_orientation_mocap_avg = ik_mocap.get_orientation_mocap(data_static_mocap_avg, cluster_use = True, task = task_static)
        cal_orientation_mocap        = ik_mocap.calibration_mocap(static_orientation_mocap_avg, cluster_use = True)


        for selected_task in task_list:
            print('*** TASK: ' + selected_task)

            try:
                print('- Obtain mocap IK')
                data_main_mocap = preprocessing_mocap.get_data_mocap(subject, selected_task)
                data_main_mocap = data_main_mocap.interpolate(method = 'cubic')
                data_main_mocap = data_main_mocap.fillna(value = 999)
                data_main_mocap = preprocessing_mocap.lowpass_filter_mocap(data_main_mocap, constant_mocap.MOCAP_SAMPLING_RATE,
                                                                           constant_mocap.FILTER_CUTOFF_MOCAP,
                                                                           constant_mocap.FILTER_ORDER) # filter
                if source == 'mt':
                    data_main_mocap = preprocessing_mocap.resample_mocap(data_main_mocap, constant_mt.MT_SAMPLING_RATE) # downsample
                elif source == 'mt_long':
                    pass # no resampling for long trial (all modalities were collected at 100 Hz)

                main_orientation_mocap = ik_mocap.get_orientation_mocap(data_main_mocap, cluster_use = True, task = selected_task)
                main_ja_mocap          = ik_mocap.get_all_ja_mocap(cal_orientation_mocap, main_orientation_mocap, cluster_use = True)

                main_ja_mocap['pelvis_tx'] = 1*(data_main_mocap['RPS2 Z'] + data_main_mocap['LPS2 Z']).to_numpy()/2
                main_ja_mocap['pelvis_ty'] = 1*(data_main_mocap['RPS2 X'] + data_main_mocap['LPS2 X']).to_numpy()/2
                main_ja_mocap['pelvis_tz'] = 1*(data_main_mocap['RPS2 Y'] + data_main_mocap['LPS2 Y']).to_numpy()/2


                if source == 'mt_long':
                    pass 
                else:
                    print('- Apply synchronization') 
                    sync_fn = f'{constant_common.OUT_SYNC_INFO}sync_info_s{subject}_{selected_task}.pkl'
                    with open(sync_fn, 'rb') as f:
                        sync_info = pickle.load(f)
                    
                    if sync_info['first_start'] == 'mocap':
                        for joint in main_ja_mocap.keys():
                            main_ja_mocap[joint] = main_ja_mocap[joint][sync_info['shifting_id']:]

                print('- Save results of ' + selected_task)
                print()
                if tuning:
                    output_path = f'outputs/mocap/tuning/{subject}/'
                    
                else:
                    if source == 'mt':
                        output_path = f'outputs/mocap/experiment_2/ik/{subject}/'
                    elif source == 'mt_long':
                        output_path = f'outputs/mocap/experiment_3/ik/{subject}/'

                common.mkfolder(output_path)
                filename = f'{output_path}{selected_task}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(main_ja_mocap, f)

            except:
                print('*** Error in processing ' + selected_task)




