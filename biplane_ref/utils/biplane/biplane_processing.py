# name: biplane_processing.py
# description: functions for processing the biplane kinematics data (e.g., resampling, interpolation, etc.)


import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd 

from constants import constant_common


def get_biplane_session_from_task(task, dataset = 'HAKnee'):

    ''' get the biplane session from the task information '''

    if dataset == 'Navio':
        
        pass # NOTE: not included in this project

    elif dataset == 'HAKnee':
        
        if task.side == 'l' and task.task == 'run':
            session = 'D'
        elif task.side == 'r' and task.task == 'run':
            session = 'C'
        elif task.side == 'l' and task.task == 'ddrop':
            session = 'B'
        else:
            session = 'A'

    return session


def get_biplane_path(subject, dataset, test, task):

    ''' get path to the biplane kinematics data '''

    subject_str = str(subject).zfill(2) 
    test_str    = 'Test' + str(test)
    session     = get_biplane_session_from_task(task)
    task_str    = task.side.upper() + constant_common.HA_TASK_MAPPING[task.task] + str(task.trial)

    path = os.path.join(constant_common.DATA_PATH, constant_common.BIPLANE_PATH, dataset, subject_str, test_str, session, task_str, 'KinematicsMeasurementReport.csv')
    
    return path


def get_biplane_knee_kinematics(subject, dataset, test, task, ft):

    ''' get biplane knee kinematics '''
    
    path = get_biplane_path(subject, dataset, test, task)

    biplane_data = pd.read_csv(path, skiprows = 1)
    biplane_data.columns.values[0] = 'Frame'
    biplane_data.columns.values[1] = 'Time'

    start_frame = biplane_data['Frame'].to_numpy()[0]
    if start_frame > 1:
        temp_frame = pd.DataFrame(np.zeros((start_frame, biplane_data.shape[1])), columns = biplane_data.columns)
        temp_frame['Frame'] = np.arange(0, start_frame)
        temp_frame['Time']  = np.arange(0, start_frame) * (1/150) # assuming the biplane data is collected at 150 Hz
        biplane_data = pd.concat([temp_frame, biplane_data], ignore_index = True)

    if task.task == 'static':
        pass
    else:
        biplane_data = biplane_resample(biplane_data, ft)

    knee_kinematics = {}
    knee_kinematics['knee_flexion_' + task.side]   = biplane_data['Flexion'].to_numpy()
    knee_kinematics['knee_adduction_' + task.side] = biplane_data['Ab\Ad'].to_numpy()
    knee_kinematics['knee_rotation_' + task.side]  = biplane_data['Int\Ext'].to_numpy()

    return knee_kinematics


def biplane_resample(biplane_data, ft):

    ''' resample biplane data to a fixed frequency '''

    ts = 1/ft 
    nan_id    = np.arange(biplane_data['Time'].to_numpy()[0], biplane_data['Time'].to_numpy()[-1], ts)
    nan_arr   = np.nan*np.ones(nan_id.shape[0])
    nan_frame = pd.DataFrame({'temp': nan_arr}, index = nan_id)

    temp_frame = biplane_data.set_index(biplane_data['Time']).iloc[:, 1::]
    temp_frame = temp_frame.join(nan_frame, how = 'outer')
    temp_frame = temp_frame.interpolate(method = 'linear', limit_area = 'inside')

    interp_frame = temp_frame.loc[nan_id, :]
    interp_frame = interp_frame.iloc[:, 0:-1]

    r_biplane_data         = 1*interp_frame.reset_index()
    r_biplane_data.columns = biplane_data.columns

    return r_biplane_data



