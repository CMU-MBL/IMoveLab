# name: eval_segment.py
# description: segment data into gait cycles or exercise reps for evaluation


import pandas as pd
import numpy as np

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mocap, constant_mt
from utils.mocap import preprocessing_mocap
from utils.events import event_mocap
from utils.eval import eval_utils


def get_event_manual(subject, task):

    ''' Get events for non-locomotion tasks from the manual segmentation '''

    event_r = []
    event_l = []

    filename = constant_common.OUT_EXERCISE_INDEX_PATH + 's' + str(subject) + '_' + 'exercise_index.xlsx'
    event_dt = pd.read_excel(filename, sheet_name = task, index_col = 0)
    
    event_r = event_dt['r'].to_numpy()
    event_l = event_dt['l'].to_numpy()

    return event_r, event_l


def get_events(subject, task, lag, fs = constant_mt.MT_SAMPLING_RATE, source = 'mt'):

    ''' Get the events for segmenting the data '''

    event = {'r': None, 'l': None}

    if task in constant_common.LIST_LOCOMOTION_TASK:
        data_main = preprocessing_mocap.get_data_mocap(subject, task)
        data_main = data_main.interpolate(method = 'cubic')
        data_main = data_main.fillna(value = 999)
        data_main = preprocessing_mocap.lowpass_filter_mocap(data_main, constant_mocap.MOCAP_SAMPLING_RATE,
                                                            constant_mocap.FILTER_CUTOFF_MOCAP,
                                                            constant_mocap.FILTER_ORDER) # filter
        data_main = preprocessing_mocap.resample_mocap(data_main, fs) # downsample
        
        mocap_traj                       = {'r': None, 'l': None}
        mocap_traj['r'], mocap_traj['l'] = event_mocap.get_marker_traj(data_main)
        event['r']                       = event_mocap.ge_heel_toe_sacrum(mocap_traj['r'], fs = fs)['hc_index']
        event['l']                       = event_mocap.ge_heel_toe_sacrum(mocap_traj['l'], fs = fs)['hc_index']

        if source == 'mt':
            sync_fn   = constant_common.OUT_SYNC_INFO + 'sync_info_s' + str(subject) + '_' + task + '.pkl'
            sync_info = eval_utils.load_data(sync_fn)

            if sync_info['first_start'] == 'mocap':
                if lag < 0:
                    shifting_id = sync_info['shifting_id'] - lag
                else:
                    shifting_id = sync_info['shifting_id']

                event['r'] = event['r'] - shifting_id 
                event['l'] = event['l'] - shifting_id 
        elif source == 'mt_long':
            print('No pre-sync for long trials')

    else:
        event['r'], event['l'] = get_event_manual(subject, task)

    return event


def get_segment(ja, event, task, fs = constant_mt.MT_SAMPLING_RATE):

    ''' Segment the data into gait cycles or exercise reps '''

    segment_ja = {}
    
    if task in constant_common.LIST_LOCOMOTION_TASK:
        min_gct = 0.3*fs
        max_gct = 1.8*fs


        for joint in ja.keys():
            if 'pelvis' in joint: # ignore pelvis rotation & translation from mocap
                continue
            
            else:
                seg_ja = []
                for i in range(len(event[joint[-1]]) - 1):
                    start = event[joint[-1]][i]
                    end   = event[joint[-1]][i + 1]

                    if (end - start) > min_gct and (end - start) < max_gct:
                        seg_ja.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, end - start), ja[joint][start:end]))

                segment_ja[joint] = np.array(seg_ja)

    else:
        for joint in ja.keys():
            if 'pelvis' in joint: # ignore pelvis rotation & translation from mocap
                continue 

            else:
                seg_ja = []

                for i in range(constant_common.NUM_EXERCISE_REPS):
                    start = event[joint[-1]][2*i]
                    end   = event[joint[-1]][2*i + 1]

                    seg_ja.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, end - start), ja[joint][start:end]))
                
                segment_ja[joint] = np.array(seg_ja)

    return segment_ja




