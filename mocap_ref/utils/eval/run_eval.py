# name: run_eval.py
# description: evaluate kinematics compared to the mocap-based reference


import numpy as np

import os, sys
sys.path.append(os.path.abspath('mocap_ref/'))

from constants import constant_common
from utils import common
from utils.eval import eval_utils, eval_segment



def evaluate(f_type, dim, subject, task, reference, mocap_alignment, selected_setup, enable_opensense = False, enable_cf = False, tuning = False, enable_psa = False, enable_drift_eval = False):

    ''' Compute the RMSE of joint angles compared to the mocap-based reference '''

    if enable_psa:
        psa_str = '_psa'
    else:
        psa_str = ''

    subject_list = common.get_subject_list(subject, tuning)
    task_list    = common.get_task_list(task)
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

                for task in task_list:
                    print('*** Task ' + task)

                    if reference == 'direct':
                        if tuning:
                            filename_mc = f'outputs/mocap/tuning/{subject}/{task}.pkl' # 11 exercises but only 2 participants
                        else:
                            filename_mc = f'outputs/mocap/experiment_2/ik/{subject}/{task}.pkl' # 11 exercises 
                        ja_mc = eval_utils.load_data(filename_mc)

                    else:
                        pass

                    if selected_setup == 'mm':
                        if tuning:
                            filter_params_str = '_'.join([str(round(param, 3)) for param in f_params])
                            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/tuning/{filter_params_str}/{subject}/{task}.pkl'
                        else:
                            # if not tuning, eval either unconstrained, sequential constraint (OpenSense), or constraint-feedback method
                            if enable_opensense: 
                                filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik_opensense/{subject}/{task}.pkl'
                                
                            elif enable_cf:
                                filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik_cf/{subject}/{task}.pkl'

                            else:
                                filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik/{subject}/{task}.pkl'
                    else:
                        if enable_cf:
                            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}/ik_cf/{subject}/{task}.pkl'
                        else:
                            filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}{psa_str}/ik/{subject}/{task}.pkl'

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


                    print('- Resync the data if lagged')
                    lag = eval_utils.find_lag(ja_mt['knee_flexion_r'], ja_mc['knee_flexion_r'])
                    if np.abs(lag) > 5:
                        lag = 0
                        
                    ja_mc, ja_mt = eval_utils.do_resync(ja_mc, ja_mt, lag)

                    
                    if mocap_alignment:
                        title_alignment = '_alignment'

                        # reverse to the original sign (i.e., different from interpretation sign)
                        for joint in ja_mt.keys():
                            ja_mt[joint] = constant_common.JA_SIGN[joint] * ja_mt[joint]
                            ja_mc[joint] = constant_common.JA_SIGN[joint] * ja_mc[joint]

                        if subject in list(constant_common.ISOLATED_CASES.keys()):
                            if task in constant_common.ISOLATED_CASES[subject].keys():
                                alignment_id = [constant_common.ISOLATED_CASES[subject][task][0], constant_common.ISOLATED_CASES[subject][task][1]]
                            else:
                                alignment_id = [constant_common.ALIGNMENT_PERIOD[0], constant_common.ALIGNMENT_PERIOD[1]]

                        else:
                            alignment_id = [constant_common.ALIGNMENT_PERIOD[0], constant_common.ALIGNMENT_PERIOD[1]]
                        
                        ja_mt = eval_utils.get_ja_alignment(ja_mt, ja_mc, alignment_id, task)
                    
                    else:
                        title_alignment = ''


                    print('- Segment the data into gait cycles or exercise reps')
                    event = eval_segment.get_events(subject, task, lag)
                    
                    segment_mc = eval_segment.get_segment(ja_mc, event, task)
                    segment_mt = eval_segment.get_segment(ja_mt, event, task)

                    print('- Evaluate the RMSE')
                    if enable_drift_eval:
                        rmse_mt = eval_utils.calculate_rmse_drift(segment_mc, segment_mt, task)
                        drift_str = 'drift_'
                    else:
                        rmse_mt = eval_utils.calculate_rmse(segment_mc, segment_mt)
                        drift_str = ''
                    print(rmse_mt)
                    
                    print('- Save the evaluation results')
                    print('\n\n\n')
                    if tuning:
                        filter_params_str = '_'.join([str(round(param, 3)) for param in f_params])
                        filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/tuning_eval/{filter_params_str}/{subject}/{task}.pkl'
                        common.mkfolder(f'outputs/{f_type.lower()}/{dim.upper()}/tuning_eval/{filter_params_str}/{subject}/')
                        eval_utils.save_data(rmse_mt, filename_mt)

                    else:
                        if enable_opensense:
                            out_folder = drift_str + 'eval_opensense'
                        elif enable_cf:
                            out_folder = drift_str + 'eval_cf'
                        else:
                            out_folder = drift_str + 'eval'
                            
                        filename_mt = f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}{psa_str}/{out_folder}/{subject}/{task}.pkl'
                        common.mkfolder(f'outputs/{f_type.lower()}/{dim.upper()}/experiment_2/{selected_setup}{psa_str}/{out_folder}/{subject}/')
                        eval_utils.save_data(rmse_mt, filename_mt)









