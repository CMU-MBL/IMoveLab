# name: synchronizer.py
# description: return sync indices for mocap and IMU data


import argparse
import pickle
import pandas as pd
import quaternion

import numpy as np
from scipy.spatial.transform import Rotation as R

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mocap, constant_mt, constant_mvn
from utils import common, synchronization
from utils.mocap import preprocessing_mocap
from utils.mt import preprocessing_mt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type = int, default = None) # subject number, if not specified, run all subjects
    parser.add_argument('--task', type = str, default = None) # task being performed, if not specified, run all tasks

    args = parser.parse_args()

    subject_list = common.get_subject_list(args.subject)
    task_list    = common.get_task_list(args.task)

    for subject in subject_list:
        print('*** Subject ' + str(subject))

        for selected_task in task_list:
            print('*** TASK: ' + selected_task)

            # try:
            print('- Obtain mocap data')
            data_mocap = preprocessing_mocap.get_data_mocap(subject, selected_task)
            data_mocap = data_mocap.fillna(value = constant_common.VERY_HIGH_NUMBER)
            data_mocap = preprocessing_mocap.lowpass_filter_mocap(data_mocap, constant_mocap.MOCAP_SAMPLING_RATE,
                                                                constant_mocap.FILTER_CUTOFF_MOCAP,
                                                                constant_mocap.FILTER_ORDER) 
            data_mocap = preprocessing_mocap.resample_mocap(data_mocap, constant_mt.MT_SAMPLING_RATE)

            print('- Obtain pelvis IMU data')
            sensor_name = 'pelvis'
            sensor_id   = constant_mt.LAB_IMU_NAME_MAP[sensor_name.upper()]
            mt_fn       = preprocessing_mt.get_data_path_mt(subject, selected_task, sensor_id)
            data_mt     = preprocessing_mt.load_data_mt(mt_fn)

            print('- Obtain sync info')
            if selected_task in ['lat_step', 'step_up_down', 'drop_jump', 'cmj', 'squat', 'step_n_hold', 'sls', 'sts']:
                iters = 600 
            else:
                iters = 1500

            first_start, shifting_id = synchronization.get_sync_info(data_mocap, data_mt, iters = iters)
            print('\n- First start: ' + str(first_start))
            print('- Shifting ID: ' + str(shifting_id) + '\n')

            print('- Save sync info of ' + selected_task)
            sync_info = {'first_start': first_start, 'shifting_id': shifting_id}
            common.mkfolder(constant_common.OUT_SYNC_INFO)
            filename = f'{constant_common.OUT_SYNC_INFO}sync_info_s{subject}_{selected_task}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(sync_info, f)
            


if __name__ == '__main__':
    main()



