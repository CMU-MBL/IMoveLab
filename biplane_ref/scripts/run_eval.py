# name: run_eval.py
# description: evaluate and obtain root-mean-square difference (RMSD) 


import pickle
import os
import numpy as np

from easydict import EasyDict

from utils import common
from utils.biplane import biplane_processing
from utils.eval import sync
from constants import constant_mc10, constant_meta
from utils.eval import alignment 


def eval_main(dataset, subject, task, trial, side, filter_type, dim, tuning = False, do_opensense = False):

    ''' Evaluation of MC10 direct (and optionally, OpenSense IK) kinematics '''

    filter_type     = filter_type.upper()
    selected_filter = filter_type

    if not tuning:
        filter_params_set = [common.get_filter_params(dataset, filter_type.lower())]

    else:
        filter_params_set = common.get_filter_params_for_tuning(filter_type)

    print('=' *50)
    print(f'Running MC10 IK with filter {selected_filter} and dim {dim} ...')

    if tuning:
        print('Tuning mode: ON')
    else:
        print('Tuning mode: OFF')

    if do_opensense:
        print('OpenSense evaluation: ON')
    else:
        print('OpenSense evaluation: OFF')

    print('=' *50)
    print()

    subject_list = common.get_subject_list(dataset, subject, tuning)
    task_list    = common.get_task_list(dataset, task)
    trial_list   = common.get_trial_list(dataset, trial)
    side_list    = common.get_side_list(dataset, side)

    for filter_params in filter_params_set:

        print(f'Filter parameters: {filter_params}\n')

        for subject in subject_list:

            for task in task_list:

                for trial in trial_list:

                    for side in side_list:

                        if constant_meta.VALID_COMPARISON[str(subject)][task][side][trial - 1]:

                            selected_task = EasyDict(side = side, trial = trial, task = task)

                            apply_sync      = True # hardcoded, always temporally sync for evaluation
                            apply_alignment = True # hardcoded, always align for evaluation
                            test            = 1 # hardcoded, only 1 test

                            mocap_fn = f'outputs/{dataset}/ik/{subject}/mocap/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                            with open(mocap_fn, 'rb') as f:
                                mocap_kinematics = pickle.load(f)
                            
                            for joint in mocap_kinematics.keys(): # set NaN values to -100
                                mocap_kinematics[joint] = mocap_kinematics[joint].fillna(-100)

                            if tuning:
                                filter_params_str = '_'.join([str(param) for param in filter_params])
                                mc10_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}/tuning/p_{filter_params_str}/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                            
                            else:
                                if do_opensense:
                                    mc10_fn_os = f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}/ik_os/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl' 
                                mc10_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim.lower()}/ik/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                            
                            with open(mc10_fn, 'rb') as f:
                                mc10_kinematics = pickle.load(f)

                            if do_opensense:
                                with open(mc10_fn_os, 'rb') as f:
                                    mc10_kinematics_os = pickle.load(f)

                            if do_opensense:
                                # hardcoded other knee DoFs as these are not available in OpenSense outputs
                                mc10_kinematics_os[f'knee_adduction_{selected_task.side}'] = np.zeros_like(mc10_kinematics_os[f'knee_flexion_{selected_task.side}'])
                                mc10_kinematics_os[f'knee_rotation_{selected_task.side}'] = np.zeros_like(mc10_kinematics_os[f'knee_flexion_{selected_task.side}'])


                            knee_kinematics = biplane_processing.get_biplane_knee_kinematics(subject, dataset, test, selected_task, ft = constant_mc10.PROCESSING_RATE)

                            # coarse temporal sync
                            if apply_sync:
                                valid_mocap_start = np.where(mocap_kinematics[f'knee_flexion_{selected_task.side}'] > -100)[0][0]
                                valid_mocap_stop  = np.where(mocap_kinematics[f'knee_flexion_{selected_task.side}'] > -100)[0][-1]
                                print(f'Mocap valid frames: {valid_mocap_start} to {valid_mocap_stop}')
                                lag = sync.find_best_match(mc10_kinematics[f'knee_flexion_{selected_task.side}'], mocap_kinematics[f'knee_flexion_{selected_task.side}'][valid_mocap_start:valid_mocap_stop], nan_flag = False) - valid_mocap_start
                                print(f'Sync lag between MC10 and Mocap: {lag} frames')

                            # trim data (hardcoded for special cases, based on data screening)
                            if subject == 6 and selected_task.task == 'shop' and selected_task.side == 'r' and selected_task.trial == 2:
                                vicon_biplane_offset = 0
                            elif subject == 12 and selected_task.task == 'shop' and selected_task.side == 'l' and selected_task.trial == 3:
                                vicon_biplane_offset = 0
                            else:
                                vicon_biplane_offset = 300
                            sync_sample_window = 20

                            # find indices of values > 0 in the biplane kinematics
                            valid_biplane_id = np.where(knee_kinematics[f'knee_flexion_{selected_task.side}'] != 0)[0]
                            # keep only the largest continuous segment
                            diff = np.diff(valid_biplane_id)
                            split_indices = np.where(diff > 1)[0] + 1
                            segments = np.split(valid_biplane_id, split_indices)
                            largest_segment = max(segments, key=len)

                            valid_biplane_start = largest_segment[1]
                            valid_biplane_stop = largest_segment[-1]

                            trimmed_knee_kinematics  = {}
                            for joint in knee_kinematics.keys():
                                trimmed_knee_kinematics[joint] = 1*knee_kinematics[joint][valid_biplane_start:valid_biplane_stop]

                            trimmed_mocap_kinematics = {}
                            for joint in mocap_kinematics.keys():
                                if vicon_biplane_offset == 0:
                                    trimmed_mocap_kinematics[joint] = 1*mocap_kinematics[joint][valid_biplane_start + vicon_biplane_offset:valid_biplane_stop + vicon_biplane_offset + sync_sample_window].to_numpy()
                                else:
                                    trimmed_mocap_kinematics[joint] = 1*mocap_kinematics[joint][valid_biplane_start + vicon_biplane_offset - sync_sample_window:valid_biplane_stop + vicon_biplane_offset + sync_sample_window].to_numpy()

                            trimmed_mc10_kinematics  = {}
                            for joint in mc10_kinematics.keys():
                                if vicon_biplane_offset == 0:
                                    trimmed_mc10_kinematics[joint] = 1*mc10_kinematics[joint][valid_biplane_start + lag + vicon_biplane_offset:valid_biplane_stop + lag + vicon_biplane_offset + sync_sample_window]
                                else:
                                    trimmed_mc10_kinematics[joint] = 1*mc10_kinematics[joint][valid_biplane_start + lag + vicon_biplane_offset - sync_sample_window:valid_biplane_stop + lag + vicon_biplane_offset + sync_sample_window]

                            if do_opensense:
                                trimmed_mc10_kinematics_os  = {}
                                for joint in mc10_kinematics_os.keys():
                                    if vicon_biplane_offset == 0:
                                        trimmed_mc10_kinematics_os[joint] = 1*mc10_kinematics_os[joint][valid_biplane_start + lag + vicon_biplane_offset:valid_biplane_stop + lag + vicon_biplane_offset + sync_sample_window]
                                    else:
                                        trimmed_mc10_kinematics_os[joint] = 1*mc10_kinematics_os[joint][valid_biplane_start + lag + vicon_biplane_offset - sync_sample_window:valid_biplane_stop + lag + vicon_biplane_offset + sync_sample_window]

                            # finer temporal sync
                            lag_mocap_biplane = sync.find_best_match(trimmed_mocap_kinematics[f'knee_flexion_{selected_task.side}'], trimmed_knee_kinematics[f'knee_flexion_{selected_task.side}'], nan_flag = False)

                            temp  = sync.find_best_match(trimmed_mc10_kinematics[f'knee_flexion_{selected_task.side}'], trimmed_knee_kinematics[f'knee_flexion_{selected_task.side}'], nan_flag = False)
                            lag_mc10_biplane  = np.min([lag_mocap_biplane + 4, temp])

                            print(f'Sync lag between Mocap and Biplane: {lag_mocap_biplane} frames')
                            print(f'Sync lag between MC10 and Biplane: {lag_mc10_biplane} frames')

                            for joint in trimmed_knee_kinematics.keys():
                                trimmed_mocap_kinematics[joint] = trimmed_mocap_kinematics[joint][lag_mocap_biplane:lag_mocap_biplane + trimmed_knee_kinematics[joint].shape[0]]
                                trimmed_mc10_kinematics[joint]  = trimmed_mc10_kinematics[joint][lag_mc10_biplane:lag_mc10_biplane + trimmed_knee_kinematics[joint].shape[0]]
                                if do_opensense:
                                    trimmed_mc10_kinematics_os[joint]  = trimmed_mc10_kinematics_os[joint][lag_mc10_biplane:lag_mc10_biplane + trimmed_knee_kinematics[joint].shape[0]]

                            # alignment before error calculation
                            if selected_task.side == 'r':
                                trimmed_mocap_kinematics[f'knee_flexion_{selected_task.side}'] *= -1

                                trimmed_mc10_kinematics[f'knee_flexion_{selected_task.side}'] *= -1

                                trimmed_knee_kinematics[f'knee_flexion_{selected_task.side}'] *= -1
                                trimmed_knee_kinematics[f'knee_adduction_{selected_task.side}'] *= -1

                                if do_opensense:
                                    trimmed_mc10_kinematics_os[f'knee_flexion_{selected_task.side}'] *= -1

                            elif selected_task.side == 'l':
                                trimmed_mocap_kinematics[f'knee_flexion_{selected_task.side}'] *= -1
                                trimmed_mocap_kinematics[f'knee_adduction_{selected_task.side}'] *= -1
                                trimmed_mocap_kinematics[f'knee_rotation_{selected_task.side}'] *= -1

                                trimmed_mc10_kinematics[f'knee_flexion_{selected_task.side}'] *= -1
                                trimmed_mc10_kinematics[f'knee_adduction_{selected_task.side}'] *= -1
                                trimmed_mc10_kinematics[f'knee_rotation_{selected_task.side}'] *= -1

                                trimmed_knee_kinematics[f'knee_flexion_{selected_task.side}'] *= -1
                                trimmed_knee_kinematics[f'knee_rotation_{selected_task.side}'] *= -1

                                if do_opensense:
                                    trimmed_mc10_kinematics_os[f'knee_flexion_{selected_task.side}'] *= -1
                                    trimmed_mc10_kinematics_os[f'knee_adduction_{selected_task.side}'] *= -1
                                    trimmed_mc10_kinematics_os[f'knee_rotation_{selected_task.side}'] *= -1

                            # SAVE_UNALIGNED: synced but NOT first-frame aligned, for per-knee offset calibration (Coupling/imu_offset_calibration.py)
                            unaligned_dir = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/eval/{subject}/unaligned/'
                            os.makedirs(unaligned_dir, exist_ok = True)
                            with open(unaligned_dir + f'knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl', 'wb') as f:
                                pickle.dump({'mc10': {k: np.asarray(v, dtype = float).copy() for k, v in trimmed_mc10_kinematics.items()},
                                             'biplane': {k: np.asarray(v, dtype = float).copy() for k, v in trimmed_knee_kinematics.items()}}, f)

                            if apply_alignment:
                                print('Applying alignment ...')

                                trimmed_mc10_kinematics  = alignment.get_ja_alignment_init(trimmed_mc10_kinematics, trimmed_knee_kinematics, selected_task.side, store_correction = False)
                                trimmed_mocap_kinematics = alignment.get_ja_alignment_init(trimmed_mocap_kinematics, trimmed_knee_kinematics, selected_task.side, store_correction = False)

                                if do_opensense:
                                    trimmed_mc10_kinematics_os = alignment.get_ja_alignment_init(trimmed_mc10_kinematics_os, trimmed_knee_kinematics, selected_task.side, store_correction = False)

                            # --- export segmented kinematics --- #
                            if not tuning:
                                if do_opensense:
                                    eval_folder = 'eval_os'
                                else:
                                    eval_folder = 'eval'

                                output_mc10_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mc10/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mc10/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mc10/')
                                with open(output_mc10_fn, 'wb') as f:
                                    if do_opensense:
                                        pickle.dump(trimmed_mc10_kinematics_os, f)
                                    else:
                                        pickle.dump(trimmed_mc10_kinematics, f)

                                output_mocap_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mocap/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mocap/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/mocap/')
                                with open(output_mocap_fn, 'wb') as f:
                                    pickle.dump(trimmed_mocap_kinematics, f)

                                output_biplane_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/biplane/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/biplane/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/biplane/')
                                with open(output_biplane_fn, 'wb') as f:
                                    pickle.dump(trimmed_knee_kinematics, f)

                            # --- calculate RMSD --- #
                            rmsd_mc10_biplane = {}
                            rmsd_mc10_mocap  = {}
                            rmsd_mocap_biplane = {}

                            for joint in knee_kinematics.keys():
                                if do_opensense:
                                    rmsd_mc10_biplane[joint] = np.sqrt(np.mean((trimmed_mc10_kinematics_os[joint] - trimmed_knee_kinematics[joint])**2))
                                    rmsd_mc10_mocap[joint]  = np.sqrt(np.mean((trimmed_mc10_kinematics_os[joint] - trimmed_mocap_kinematics[joint])**2))
                                    rmsd_mocap_biplane[joint] = np.sqrt(np.mean((trimmed_mocap_kinematics[joint] - trimmed_knee_kinematics[joint])**2))
                                else:
                                    rmsd_mc10_biplane[joint] = np.sqrt(np.mean((trimmed_mc10_kinematics[joint] - trimmed_knee_kinematics[joint])**2))
                                    rmsd_mc10_mocap[joint]  = np.sqrt(np.mean((trimmed_mc10_kinematics[joint] - trimmed_mocap_kinematics[joint])**2))
                                    rmsd_mocap_biplane[joint] = np.sqrt(np.mean((trimmed_mocap_kinematics[joint] - trimmed_knee_kinematics[joint])**2))
                                
                                print(f'RMSD {joint} - MC10: {rmsd_mc10_biplane[joint]:.2f}, Mocap: {rmsd_mocap_biplane[joint]:.2f}')
                                print(f'MC10 with Mocap: {rmsd_mc10_mocap[joint]:.2f}')
                                print('---')
                                

                            if tuning:
                                filter_params_str = '_'.join([str(param) for param in filter_params])
                                output_rmsd_mc10_biplane_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}//tuning_eval/p_{filter_params_str}/{subject}/rmsd_mc10_biplane_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/')
                            else:
                                output_rmsd_mc10_biplane_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/rmsd_mc10_biplane_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/')
                            with open(output_rmsd_mc10_biplane_fn, 'wb') as f:
                                pickle.dump(rmsd_mc10_biplane, f)

                            if tuning:
                                filter_params_str = '_'.join([str(param) for param in filter_params])
                                output_rmsd_mc10_mocap_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}//tuning_eval/p_{filter_params_str}/{subject}/rmsd_mc10_mocap_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/')
                            else:
                                output_rmsd_mc10_mocap_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/rmsd_mc10_mocap_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/')
                            with open(output_rmsd_mc10_mocap_fn, 'wb') as f:
                                pickle.dump(rmsd_mc10_mocap, f)

                            if tuning:
                                filter_params_str = '_'.join([str(param) for param in filter_params])
                                output_rmsd_mocap_biplane_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}//tuning_eval/p_{filter_params_str}/{subject}/rmsd_mocap_biplane_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/tuning_eval/p_{filter_params_str}/{subject}/')
                            else:
                                output_rmsd_mocap_biplane_fn = f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/rmsd_mocap_biplane_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'
                                if not os.path.exists(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/'):
                                    os.makedirs(f'outputs/{dataset}/bm_{filter_type.lower()}{dim}/{eval_folder}/{subject}/')
                            with open(output_rmsd_mocap_biplane_fn, 'wb') as f:
                                pickle.dump(rmsd_mocap_biplane, f)









