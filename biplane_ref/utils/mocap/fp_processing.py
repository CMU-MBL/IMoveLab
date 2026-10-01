# name: fp_processing.py
# description: processing functions for the force plate data


import ezc3d

import pandas as pd
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mocap import mocap_processing



def get_fp_data(dataset, subject, test, task):

    ''' get fp data from c3d file '''
    
    path = mocap_processing.get_mocap_path(dataset, subject, test, task)

    c3d_data = ezc3d.c3d(path)

    fp_data = {}

    fp_list          = c3d_data['parameters']['ANALOG']['LABELS']['value']
    selected_fp_list = [marker for marker in fp_list if "Force" in marker]

    # get sampling rate
    fs = c3d_data['parameters']['ANALOG']['RATE']['value'][0]
    print(f'FP sampling rate: {fs} Hz')

    for fp in selected_fp_list:
        fp_data[fp] = (-1) * c3d_data['data']['analogs'][:, fp_list.index(fp), :].T

        fp_data[fp] = pd.DataFrame(
            fp_data[fp], 
            columns=[f"{fp.split('.')[1]}"]
        )

    fp_data = pd.concat(fp_data.values(), axis = 1)

    num_frames = fp_data.shape[0]
    time       = np.arange(0, num_frames/fs, 1/fs)
    time       = pd.DataFrame(time, columns = ['Time'])

    fp_data = pd.concat([time, fp_data], axis = 1)


    return fp_data





