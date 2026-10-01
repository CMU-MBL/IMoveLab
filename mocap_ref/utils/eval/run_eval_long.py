# name: run_eval_long.py
# description: evaluate kinematics compared to the mocap-based reference


import numpy as np
import pandas as pd

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common, constant_mocap, constant_mt
from utils import common
from utils.mocap import ik_mocap
from utils.eval import eval_utils, eval_segment
from utils.eval import metrics



def evaluate(f_type, dim, subject, reference, mocap_alignment, selected_setup, enable_opensense = False, enable_cf = False, eval_mode = 'trial_rmsd', enable_mocap = False):

    ''' Obtain RMSD during long-duration trials '''
    
    print('*** Filter ' + f_type)
    print('*** Sensor axes ' + dim.upper())

    task = 'long_walk'

    subject_list = common.get_subject_list_long(subject)

    for subject in subject_list:
        print('*** Subject ' + str(subject))

        # use 9D data for sync
        if f_type.lower() == 'riann':
            filename_mt_9d = f'outputs/{f_type.lower()}/6D/experiment_3/{selected_setup}/ik/{subject}/{task}1.pkl' # RIANN doesn't have 9D
        else:
            filename_mt_9d = f'outputs/{f_type.lower()}/9D/experiment_3/mm/ik/{subject}/{task}1.pkl'
        ja_mt_9d = eval_utils.load_data(filename_mt_9d)

        if enable_opensense:
            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/ik_opensense/{subject}/{task}1.pkl'
            out_folder  = 'eval_opensense'
        elif enable_cf:
            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/ik_cf/{subject}/{task}1.pkl'
            out_folder  = 'eval_cf'
        else:
            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/ik/{subject}/{task}1.pkl'
            out_folder  = 'eval'

        ja_mt = eval_utils.load_data(filename_mt)

        if enable_opensense:
            rename_keys = {'ankle_angle_r': 'ankle_flexion_r', 'ankle_angle_l': 'ankle_flexion_l'}
            for old_key, new_key in rename_keys.items():
                ja_mt[new_key] = ja_mt.pop(old_key)

            # XXX: hardcoded as these are not available in OpenSense outputs
            ja_mt['knee_adduction_r']  = np.zeros(ja_mt['knee_flexion_r'].shape)
            ja_mt['knee_adduction_l']  = np.zeros(ja_mt['knee_flexion_l'].shape)
            ja_mt['knee_rotation_r']   = np.zeros(ja_mt['knee_flexion_r'].shape)
            ja_mt['knee_rotation_l']   = np.zeros(ja_mt['knee_flexion_l'].shape)
            ja_mt['ankle_adduction_r'] = np.zeros(ja_mt['ankle_flexion_r'].shape)
            ja_mt['ankle_adduction_l'] = np.zeros(ja_mt['ankle_flexion_l'].shape)
            ja_mt['ankle_rotation_r']  = np.zeros(ja_mt['ankle_flexion_r'].shape)
            ja_mt['ankle_rotation_l']  = np.zeros(ja_mt['ankle_flexion_l'].shape)
        
        mocap_t1_fn = f'outputs/mocap/experiment_3/ik/{subject}/{task}1.pkl'
        mocap_t2_fn = f'outputs/mocap/experiment_3/ik/{subject}/{task}2.pkl'
        mocap_t3_fn = f'outputs/mocap/experiment_3/ik/{subject}/{task}3.pkl'
        
        ja_mc_trial       = {}
        ja_mc_trial['t1'] = eval_utils.load_data(mocap_t1_fn)
        ja_mc_trial['t2'] = eval_utils.load_data(mocap_t2_fn)
        ja_mc_trial['t3'] = eval_utils.load_data(mocap_t3_fn)


        if eval_mode == 'visualization':
            pass
        
        else:
            # reverse to the original sign (i.e., different from interpretation sign)
            for joint in ja_mt.keys():
                ja_mt[joint]    = constant_common.JA_SIGN[joint] * ja_mt[joint]
                ja_mt_9d[joint] = constant_common.JA_SIGN[joint] * ja_mt_9d[joint]

                ja_mc_trial['t1'][joint] = constant_common.JA_SIGN[joint] * ja_mc_trial['t1'][joint]
                ja_mc_trial['t2'][joint] = constant_common.JA_SIGN[joint] * ja_mc_trial['t2'][joint]
                ja_mc_trial['t3'][joint] = constant_common.JA_SIGN[joint] * ja_mc_trial['t3'][joint]


        if mocap_alignment:
            title_alignment = '_alignment'

            if subject in list(constant_common.ISOLATED_CASES.keys()):
                if task in constant_common.ISOLATED_CASES[subject].keys():
                    alignment_id = [constant_common.ISOLATED_CASES[subject][task][0], constant_common.ISOLATED_CASES[subject][task][1]]
                else:
                    alignment_id = [constant_common.ALIGNMENT_PERIOD[0], constant_common.ALIGNMENT_PERIOD[1]]

            else:
                alignment_id = [constant_common.ALIGNMENT_PERIOD[0], constant_common.ALIGNMENT_PERIOD[1]]
            
            if eval_mode == 'visualization':
                print('*** Using neutral pose for alignment (for visualization mode) ***')
                neurtral_pose = {}
                for joint in ja_mt.keys():
                    neurtral_pose[joint] = np.zeros(ja_mt[joint].shape[0])

                ja_mc_trial['t1'] = eval_utils.get_ja_alignment(ja_mc_trial['t1'], neurtral_pose, alignment_id, task, store_correction = True)
                ja_mc_trial['t2'] = eval_utils.get_ja_alignment(ja_mc_trial['t2'], neurtral_pose, alignment_id, task)
                ja_mc_trial['t3'] = eval_utils.get_ja_alignment(ja_mc_trial['t3'], neurtral_pose, alignment_id, task)

            ja_mt    = eval_utils.get_ja_alignment(ja_mt, ja_mc_trial['t1'], alignment_id, task)
            ja_mt_9d = eval_utils.get_ja_alignment(ja_mt_9d, ja_mc_trial['t1'], alignment_id, task)
        
        else:
            title_alignment = ''

        ja_mt_trial       = {}
        ja_mt_trial['t1'] = eval_utils.get_long_trial_chunk(ja_mt, subject, 1)
        ja_mt_trial['t2'] = eval_utils.get_long_trial_chunk(ja_mt, subject, 2)
        ja_mt_trial['t3'] = eval_utils.get_long_trial_chunk(ja_mt, subject, 3)

        ja_mt_trial_9d = {}
        ja_mt_trial_9d['t1'] = eval_utils.get_long_trial_chunk(ja_mt_9d, subject, 1)
        ja_mt_trial_9d['t2'] = eval_utils.get_long_trial_chunk(ja_mt_9d, subject, 2)
        ja_mt_trial_9d['t3'] = eval_utils.get_long_trial_chunk(ja_mt_9d, subject, 3)

        for trial in ['t1', 't2', 't3']:
            lag = eval_utils.find_lag(ja_mt_trial_9d[trial]['knee_flexion_r'], ja_mc_trial[trial]['knee_flexion_r'])
        
            ja_mc_trial[trial], ja_mt_trial[trial] = eval_utils.do_resync(ja_mc_trial[trial], ja_mt_trial[trial], lag)
    
            if eval_mode != 'visualization':
                ja_mc_trial[trial], ja_mt_trial[trial] = eval_utils.remove_bad_mocap(ja_mc_trial[trial], ja_mt_trial[trial], subject, trial) # --> disable this when outputing motion files


        # 1. Get RMSDs of three trials
        if eval_mode == 'trial_rmsd':
            for trial in ['t1', 't2', 't3']:
                event = eval_segment.get_events(subject, task + str(trial[-1]), lag, fs = constant_mocap.MOCAP_SAMPLING_RATE, source = 'mt_long')
                for side in event.keys():
                    id = np.where(event[side] < constant_mt.LONG_TRIAL_ID[subject][trial]['sitting_start'])[0]
                    event[side] = event[side][id]

                segment_mc = eval_segment.get_segment(ja_mc_trial[trial], event, task + str(trial[-1]), fs = constant_mocap.MOCAP_SAMPLING_RATE)
                segment_mt = eval_segment.get_segment(ja_mt_trial[trial], event, task + str(trial[-1]), fs = constant_mocap.MOCAP_SAMPLING_RATE)

                rmsd_mt = eval_utils.calculate_rmse(segment_mc, segment_mt)
                print(rmsd_mt)

                filename = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/{task}{trial[-1]}.pkl'
                common.mkfolder(f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/')
                print(filename)
                eval_utils.save_data(rmsd_mt, filename)
                print('DONE')

        # 2. Get the cumulative RMSDs (to show drifting)
        elif eval_mode == 'minute_rmsd':
            for trial in ['t1', 't2', 't3']:
                print('- Trial ' + str(trial))
                event = eval_segment.get_events(subject, task + str(trial[-1]), lag, fs = constant_mocap.MOCAP_SAMPLING_RATE, source = 'mt_long')
                for side in event.keys():
                    id = np.where(event[side] < constant_mt.LONG_TRIAL_ID[subject][trial]['sitting_start'])[0]
                    event[side] = event[side][id]

                rmsd_mt = {}
                for joint in ja_mt_trial[trial].keys():
                    rmsd_mt[joint] = []
                    if '_r' in joint:
                        start_id = event['r'][0]
                        end_id   = event['r'][-1]
                    else:
                        start_id = event['l'][0]
                        end_id   = event['l'][-1]
                    
                    chunk = 1*start_id
                    while (chunk < end_id) and (chunk < 60000):
                        if chunk + 6000 < end_id:
                            rmsd_mt[joint].append(metrics.get_rmse(ja_mc_trial[trial][joint][chunk:chunk + 6000], ja_mt_trial[trial][joint][chunk:chunk + 6000]))
                        else:
                            if end_id > 60000:
                                rmsd_mt[joint].append(metrics.get_rmse(ja_mc_trial[trial][joint][chunk:60000], ja_mt_trial[trial][joint][chunk:60000]))
                            else:
                                rmsd_mt[joint].append(metrics.get_rmse(ja_mc_trial[trial][joint][chunk:end_id], ja_mt_trial[trial][joint][chunk:end_id]))

                        chunk += 6000

                print(rmsd_mt)
                # filename = 'outputs/rmse_longwalk/s' + subject + '_' + f_type.lower() + '_' + dim.upper() + '_' + task + str(trial[-1]) + '_mt_chunk.pkl'
                filename = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/{task}{trial[-1]}.pkl'
                common.mkfolder(f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/')
                # print(filename)
                eval_utils.save_data(rmsd_mt, filename)
                print('DONE')
            
        # 3. Get IK for OpenSense
        elif eval_mode == 'visualization':
            for trial in ['t1', 't2', 't3']:
                print('- Trial ' + str(trial))

                if enable_mocap:
                    time_id = np.arange(0, len(ja_mc_trial[trial]['knee_flexion_r']), 1)/constant_mocap.MOCAP_SAMPLING_RATE
                else:
                    time_id = np.arange(0, len(ja_mt_trial[trial]['knee_flexion_r']), 1)/constant_mocap.MOCAP_SAMPLING_RATE
                

                output = pd.DataFrame()
                output['time'] = time_id
                
                if enable_mocap:
                    print('*** Outputing mocap-based motion file (for visualization) ***')
                    # NOTE: for mocap visualization
                    output['hip_adduction_r'] = ja_mc_trial[trial]['hip_adduction_r']
                    output['hip_rotation_r']  = ja_mc_trial[trial]['hip_rotation_r']
                    output['hip_flexion_r']   = ja_mc_trial[trial]['hip_flexion_r']
                    output['knee_angle_r']    = ja_mc_trial[trial]['knee_flexion_r']
                    output['ankle_angle_r']   = ja_mc_trial[trial]['ankle_flexion_r']

                    output['hip_adduction_l'] = ja_mc_trial[trial]['hip_adduction_l']
                    output['hip_rotation_l']  = ja_mc_trial[trial]['hip_rotation_l']
                    output['hip_flexion_l']   = ja_mc_trial[trial]['hip_flexion_l']
                    output['knee_angle_l']    = ja_mc_trial[trial]['knee_flexion_l']
                    output['ankle_angle_l']   = ja_mc_trial[trial]['ankle_flexion_l']
                    out_fn = f'outputs/mocap/experiment_3/eval/{eval_mode}/{subject}/{task}{trial[-1]}.mot'
                    common.mkfolder(f'outputs/mocap/experiment_3/eval/{eval_mode}/{subject}/')

                else:
                    print('*** Outputing IMU-based motion file (for visualization) ***')
                    # NOTE: for IMU visualization
                    output['hip_adduction_r'] = ja_mt_trial[trial]['hip_adduction_r']
                    output['hip_rotation_r']  = ja_mt_trial[trial]['hip_rotation_r']
                    output['hip_flexion_r']   = ja_mt_trial[trial]['hip_flexion_r']
                    output['knee_angle_r']    = ja_mt_trial[trial]['knee_flexion_r']
                    output['ankle_angle_r']   = ja_mt_trial[trial]['ankle_flexion_r']

                    output['hip_adduction_l'] = ja_mt_trial[trial]['hip_adduction_l']
                    output['hip_rotation_l']  = ja_mt_trial[trial]['hip_rotation_l']
                    output['hip_flexion_l']   = ja_mt_trial[trial]['hip_flexion_l']
                    output['knee_angle_l']    = ja_mt_trial[trial]['knee_flexion_l']
                    output['ankle_angle_l']   = ja_mt_trial[trial]['ankle_flexion_l']
                    out_fn = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/{task}{trial[-1]}.mot'
                    common.mkfolder(f'outputs/{f_type.lower()}/{dim.upper()}/experiment_3/{selected_setup}/{out_folder}/{eval_mode}/{subject}/')

                
                prefix = 'inDegrees=yes\n'
                prefix += 'name=ik_imu_orientation\n'
                prefix += 'DataType=double\n'
                prefix += 'version=3\n'
                prefix += 'OpenSimVersion=4.5-2023-11-26-efcdfd3eb\n'
                prefix += 'endheader\n'

                output.to_csv(out_fn, sep = '\t', index = False)
                with open(out_fn, 'r') as original: data = original.read()
                with open(out_fn, 'w') as modified: modified.write(prefix + data)

        # print('DONE')











