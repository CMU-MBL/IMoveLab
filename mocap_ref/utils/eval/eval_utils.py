# name: eval_utils.py
# description: utility functions for evaluation


import pickle as pkl
import numpy as np

from scipy import signal
from scipy.spatial.transform import Rotation as R

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from utils.eval import metrics
from constants import constant_common, constant_mt


def load_data(filename):

    ''' Load data from a pickle file '''

    with open(filename, 'rb') as f:
        data = pkl.load(f)

    return data


def save_data(data, filename):

    ''' Save data to a pickle file '''

    with open(filename, 'wb') as f:
        pkl.dump(data, f)


def find_lag(signal1, signal2):

    ''' Find the lag between two signals '''

    corr = np.correlate(signal1, signal2, mode = 'full')
    lags = signal.correlation_lags(len(signal1), len(signal2))
    lag  = lags[np.argmax(abs(corr))]

    return lag


def resync_data(ja, lag):

    ''' Resync the data based on the lag '''

    for joint in ja.keys():
        ja[joint] = ja[joint][lag:]

    return ja


def do_resync(ja_mc, ja_mt, lag):

    ''' Resync the data based on the lag '''

    if lag > 0:
        ja_mt = resync_data(ja_mt, lag)

    elif lag < 0:
        ja_mc = resync_data(ja_mc, -lag)

    return ja_mc, ja_mt


def do_resync_mvn(ja_mc, ja_mvn, ja_mvn_opensense, ja_mvn_biomodel, lag):

    ''' Resync the data based on the lag '''

    if lag > 0:
        ja_mvn           = resync_data(ja_mvn, lag)
        ja_mvn_opensense = resync_data(ja_mvn_opensense, lag)
        ja_mvn_biomodel  = resync_data(ja_mvn_biomodel, lag)

    elif lag < 0:
        ja_mc = resync_data(ja_mc, -lag)

    return ja_mc, ja_mvn, ja_mvn_opensense, ja_mvn_biomodel


def calculate_rmse(segment_mc, segment_imu, do_biomodel = False):

    ''' Calculate the root-mean-square error (RMSE) '''

    rmse_imu = {}

    for joint in segment_mc.keys():
        if do_biomodel:
            if joint[:-2] in ['knee_adduction', 'knee_rotation', 'ankle_adduction', 'ankle_rotation']: # NOTE: redundant since zeros are already assigned to these joints when using OpenSense
                continue
            else:
                rmse_imu[joint] = metrics.get_rmse(segment_mc[joint].flatten(), segment_imu[joint].flatten())
        else:
            rmse_imu[joint] = metrics.get_rmse(segment_mc[joint].flatten(), segment_imu[joint].flatten())
    
    return rmse_imu


def calculate_rmse_drift(segment_mc, segment_mt, task):

    ''' Calculate the root-mean-square error (RMSE) of the first (few) cycle(s) and the last (few) cycle(s) to show drifting '''

    rmse_imu = {'start': {}, 'end': {}}

    for joint in segment_mc.keys():
        if task in constant_common.LIST_LOCOMOTION_TASK:
            rmse_imu['start'][joint] = metrics.get_rmse(segment_mc[joint][:5].flatten(), segment_mt[joint][:5].flatten())
            rmse_imu['end'][joint]   = metrics.get_rmse(segment_mc[joint][-5:].flatten(), segment_mt[joint][-5:].flatten())

        else:
            rmse_imu['start'][joint] = metrics.get_rmse(segment_mc[joint][0].flatten(), segment_mt[joint][0].flatten())
            rmse_imu['end'][joint]   = metrics.get_rmse(segment_mc[joint][-1].flatten(), segment_mt[joint][-1].flatten())

    return rmse_imu


def get_ja_alignment(ja_mt, ja_mc, alignment_id, task, store_correction = False):

    ''' Get the joint angles alignment '''

    sequence = 'ZXY'

    aligned_ja_mt  = {}

    for side in ['r', 'l']:
        for joint in ['hip', 'knee', 'ankle']:

            if 'treadmill' in task and side == 'l':
                # no alignment for the left side during treadmill locomotion (due to the issue in the mocap data)
                aligned_ja_mt[joint + '_flexion_' + side] = ja_mt[joint + '_flexion_' + side]
                aligned_ja_mt[joint + '_adduction_' + side] = ja_mt[joint + '_adduction_' + side]
                aligned_ja_mt[joint + '_rotation_' + side] = ja_mt[joint + '_rotation_' + side]
                
            else:
                angle_mt     = np.array([ja_mt[joint + '_flexion_' + side], 
                                        ja_mt[joint + '_adduction_' + side], 
                                        ja_mt[joint + '_rotation_' + side]]).T
                rot_angle_mt = R.from_euler(sequence, angle_mt, degrees = True).as_matrix()

                static_angle_mt     = np.array([ja_mt[joint + '_flexion_' + side][alignment_id[0]:alignment_id[1]].mean(),
                                                ja_mt[joint + '_adduction_' + side][alignment_id[0]:alignment_id[1]].mean(),
                                                ja_mt[joint + '_rotation_' + side][alignment_id[0]:alignment_id[1]].mean()]).T
                rot_static_angle_mt = R.from_euler(sequence, static_angle_mt, degrees = True).as_matrix()

                static_angle_mocap     = np.array([ja_mc[joint + '_flexion_' + side][alignment_id[0]:alignment_id[1]].mean(),
                                                    ja_mc[joint + '_adduction_' + side][alignment_id[0]:alignment_id[1]].mean(),
                                                    ja_mc[joint + '_rotation_' + side][alignment_id[0]:alignment_id[1]].mean()]).T  
                rot_static_angle_mocap = R.from_euler(sequence, static_angle_mocap, degrees = True).as_matrix()

                correction = rot_static_angle_mt.T @ rot_static_angle_mocap

                aligned_angle_mt = np.zeros(rot_angle_mt.shape)
                for i in range(rot_angle_mt.shape[0]):
                    aligned_angle_mt[i] = rot_angle_mt[i] @ correction
                
                rot_aligned_angle_mt = R.from_matrix(aligned_angle_mt).as_euler(sequence, degrees = True)
                aligned_ja_mt[joint + '_flexion_' + side] = rot_aligned_angle_mt[:, 0]
                aligned_ja_mt[joint + '_adduction_' + side] = rot_aligned_angle_mt[:, 1]
                aligned_ja_mt[joint + '_rotation_' + side] = rot_aligned_angle_mt[:, 2]

    if store_correction:
        return aligned_ja_mt, correction
    
    else:
        return aligned_ja_mt


def get_long_trial_chunk(ja_mt, subject, trial):

    ''' Get the chunk of the long trial based on the identified trial start and end '''

    ja_mt_trial = {}
    for joint in ja_mt.keys():
        ja_mt_trial[joint] = 1*ja_mt[joint][constant_mt.LONG_TRIAL_ID[subject]['t' + str(trial)]['trial_start']:constant_mt.LONG_TRIAL_ID[subject]['t' + str(trial)]['trial_end']]

    return ja_mt_trial


def remove_bad_mocap(ja_mc, ja_mt, subject, trial):

    ''' Remove the bad mocap data based on the identified bad segments '''

    for i in range(len(constant_mt.REMOVAL_OF_BAD_MOCAP[subject][trial])):
        if len(constant_mt.REMOVAL_OF_BAD_MOCAP[subject][trial][i]) > 0:
            for joint in ja_mt.keys(): # use ja_mt instead because ja_mc may contain pelvis rotation & translation
                start_id = constant_mt.REMOVAL_OF_BAD_MOCAP[subject][trial][i][0]
                end_id   = constant_mt.REMOVAL_OF_BAD_MOCAP[subject][trial][i][1]
                ja_mc[joint][start_id:end_id] = np.nan
                ja_mt[joint][start_id:end_id] = np.nan

    return ja_mc, ja_mt



