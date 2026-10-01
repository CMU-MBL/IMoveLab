# name: run_mocap.py
# description: run IK for the mocap data (Vicon)


import os
import pickle

from easydict import EasyDict

from constants import constant_mocap, constant_mc10, constant_meta
from utils.mocap import mocap_processing, mocap_ik, fp_processing
from utils import common

import matplotlib.pyplot as plt


# --- Kinematics from c3d file --- #
dataset = 'HAKnee'


def mocap_ik_main(dataset, subject, task, trial, side):

    subject_list = common.get_subject_list(dataset, subject)
    task_list    = common.get_task_list(dataset, task)
    trial_list   = common.get_trial_list(dataset, trial)
    side_list    = common.get_side_list(dataset, side)


    for subject in subject_list:

        print(f'*** Subject {subject}')
        test = 1

        for task in task_list:
            print(f'--- Task {task}')

            for trial in trial_list:

                for side in side_list:

                    try:
                        task_static = EasyDict(side = constant_meta.STATIC_SIDE[str(subject)], trial = 1, task = 'static')
                        
                        selected_task = EasyDict(side = side, trial = trial, task = task)

                        mocap_data_static, marker_list = mocap_processing.get_mocap_data(dataset, subject, test, task_static)
                        mocap_data_static              = mocap_data_static.interpolate(method = 'linear', limit_area = 'inside')
                        mocap_data_static              = mocap_data_static.bfill().ffill() # fill NaN with the previous and next valid value
                        mocap_data_static_mean         = mocap_processing.get_average_mocap(mocap_data_static)

                        fp_data = fp_processing.get_fp_data(dataset, subject, test, selected_task)
                        fp_data = mocap_processing.mocap_resample(fp_data, constant_mc10.PROCESSING_RATE)

                        static_orientation_mocap_mean = mocap_ik.get_orientation_mocap(mocap_data_static_mean, tracking = True, task = task_static.task)
                        cal_orientation_mocap         = mocap_ik.calibrate_mocap(static_orientation_mocap_mean, tracking = True)

                        mocap_data_main, _ = mocap_processing.get_mocap_data(dataset, subject, test, selected_task)
                        mocap_data_main    = mocap_data_main.interpolate(method = 'linear', limit_area = 'inside')
                        mocap_data_main    = mocap_processing.mocap_resample(mocap_data_main, constant_mc10.PROCESSING_RATE) # processed at 150 Hz (no need to resample)

                        mocap_masking      = mocap_processing.get_mocap_masking(mocap_data_main, dataset) # get the mask of the valid mocap data
                        mocap_data_main    = mocap_data_main.bfill().ffill() # fill NaN with the previous and next valid value

                        mocap_data_main    = mocap_processing.mocap_lowpass_filter(mocap_data_main, constant_mc10.PROCESSING_RATE, constant_mocap.FILTER_CUTOFF_MOCAP, fo = 4)


                        main_orientation_mocap = mocap_ik.get_orientation_mocap(mocap_data_main, tracking = True, task = selected_task.task)
                        knee_kinematics        = mocap_ik.get_knee_kinematics(cal_orientation_mocap, main_orientation_mocap, tracking = True)


                        knee_kinematics_masked = {}
                        for joint in knee_kinematics.keys():
                            knee_kinematics_masked[joint] = knee_kinematics[joint] * mocap_masking

                        # export the kinematics to a pickle file
                        output_fn = f'outputs/{dataset}/ik/{subject}/mocap/knee_kinematics_{selected_task.side}_{selected_task.task}_{selected_task.trial}.pkl'

                        if not os.path.exists(f'outputs/{dataset}/ik/{subject}/mocap/'):
                            os.makedirs(f'outputs/{dataset}/ik/{subject}/mocap/')

                        with open(output_fn, 'wb') as f:
                            pickle.dump(knee_kinematics_masked, f)


                        fig, ax = plt.subplots(4, 2, figsize = (10, 5), sharex = True)

                        ax[0, 0].plot(knee_kinematics['knee_flexion_r'], color = 'gray', linestyle = ':', label = 'Interpolated data')
                        ax[0, 0].plot(knee_kinematics_masked['knee_flexion_r'], color = 'r', lw = 2, label = 'Valid data')
                        ax[0, 0].set_ylabel(r'KF ($^\circ$)')
                        ax[0, 0].legend(frameon = False, fontsize = 10, ncol = 2)
                        ax[0, 0].set_xlim(0, knee_kinematics['knee_flexion_r'].shape[0])
                        ax[1, 0].plot(knee_kinematics['knee_adduction_r'], color = 'gray', linestyle = ':')
                        ax[1, 0].plot(knee_kinematics_masked['knee_adduction_r'], color = 'r', lw = 2)
                        ax[1, 0].set_ylabel(r'KA ($^\circ$)')
                        ax[2, 0].plot(knee_kinematics['knee_rotation_r'], color = 'gray', linestyle = ':')
                        ax[2, 0].plot(knee_kinematics_masked['knee_rotation_r'], color = 'r', lw = 2)
                        ax[2, 0].set_ylabel(r'KR ($^\circ$)')
                        ax[3, 0].plot(fp_data['Fz2'], color = 'gray', linestyle = '-')
                        # ax[3, 0].plot(fp_data['Fz1'] * mocap_masking, color = 'r', lw = 2)
                        ax[3, 0].set_ylabel('GRF (N)')
                        ax[3, 0].set_xlabel('Time step')

                        ax[0, 1].plot(knee_kinematics['knee_flexion_l'], color = 'gray', linestyle = ':')
                        ax[0, 1].plot(knee_kinematics_masked['knee_flexion_l'], color = 'r', lw = 2)
                        ax[1, 1].plot(knee_kinematics['knee_adduction_l'], color = 'gray', linestyle = ':')
                        ax[1, 1].plot(knee_kinematics_masked['knee_adduction_l'], color = 'r', lw = 2)
                        ax[2, 1].plot(knee_kinematics['knee_rotation_l'], color = 'gray', linestyle = ':')
                        ax[2, 1].plot(knee_kinematics_masked['knee_rotation_l'], color = 'r', lw = 2)
                        ax[3, 1].plot(fp_data['Fz1'], color = 'gray', linestyle = '-')
                        # ax[3, 1].plot(fp_data['Fz2'] * mocap_masking, color = 'r', lw = 2)
                        ax[3, 1].set_xlabel('Time step')

                        for i in range(4):
                            for j in range(2):
                                ax[i, j].spines['top'].set_visible(False)
                                ax[i, j].spines['right'].set_visible(False)

                        for i in range(3):
                            for j in range(2):
                                ax[i, j].set_ylim(-25, 100)


                        path = f'figures/ik/mocap/{dataset}/subject{subject}/'
                        os.makedirs(path, exist_ok = True)

                        plt.savefig(path + f'{selected_task.side}_{selected_task.task}_{selected_task.trial}.svg')
                        plt.close()
                        
                    
                    except:

                        print(f'No data for subject {subject}, trial {trial}')









