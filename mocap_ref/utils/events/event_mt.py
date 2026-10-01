# name: event_mt.py
# description: get gait or repetitive events from IMU data


import numpy as np
import pandas as pd

from scipy import signal


# --- Filter data before identifying gait cycles or exercise repetitions --- #
def filter_signal(input_signal, fs, cutoff = 15):

    ''' Filter data before identifying gait cycles or exercise repetitions '''

    Wn   = cutoff*2/fs 
    b, a = signal.butter(4, Wn, btype = 'low')

    filtered_signal = signal.filtfilt(b, a, input_signal)

    return filtered_signal


# --- Detect exercise repetition --- #
def detect_exercise_rep(input_signal, fs = 40):

    ''' Obtain index of repetitions of an exercise '''

    min_peak_distance = 0.5*fs 

    filtered_signal = filter_signal(input_signal, fs, cutoff = 5)
    peak_index, _   = signal.find_peaks(filtered_signal, height = 2, distance = min_peak_distance)
    threshold       = 2*fs 
    peak_to_peak    = 0
    peak_squat_id   = 0
    while peak_to_peak < threshold:
        peak_to_peak  = peak_index[peak_squat_id + 1] - peak_index[peak_squat_id]
        peak_squat_id = peak_squat_id + 1
    peak_index      = peak_index[peak_squat_id:(peak_squat_id + 4)]
    zero_crossings  = np.where(np.diff(np.sign(filtered_signal)))[0]

    start_index = zero_crossings[np.where(zero_crossings < peak_index[0])[0][-1]]
    stop_index  = zero_crossings[np.where(zero_crossings > peak_index[-1])[0][1]]

    rep_events = {'start_index': start_index, 'stop_index': stop_index}

    return rep_events



# --- Detect gait events during walking --- #
def detect_gait_events(lateral_shank_vel, fs = 40):

    ''' Obtain gait events, including heel contact, toe off, and mid swing from IMU data '''

    min_peak_distance = 0.5*fs 
    
    filtered_lateral_shank_vel = filter_signal(lateral_shank_vel, fs)

    mid_swing_index, mid_swing_value = signal.find_peaks(filtered_lateral_shank_vel, height = [2, 10], distance = min_peak_distance)
    mid_swing_value                  = mid_swing_value['peak_heights']
    stance_index, stance_value       = signal.find_peaks(-1*filtered_lateral_shank_vel, height = -1)

    gait_events = {'hc_index': [], 'hc_value': [], 'to_index': [], 'to_value': []}

    for id in mid_swing_index:
        temp_id = np.where(stance_index > id)[0][0]
        gait_events['hc_index'].append(stance_index[temp_id])
        gait_events['hc_value'].append(-1*stance_value['peak_heights'][temp_id])

        temp_id = np.where(stance_index < id)[0][-1]
        if stance_value['peak_heights'][temp_id-1] > stance_value['peak_heights'][temp_id]:
            if (stance_index[temp_id] - stance_index[temp_id - 1]) < 0.25*fs:
                temp_id = temp_id - 1
        gait_events['to_index'].append(stance_index[temp_id])
        gait_events['to_value'].append(-1*stance_value['peak_heights'][temp_id])

    gait_events['hc_index'] = np.array(gait_events['hc_index'])
    gait_events['hc_value'] = np.array(gait_events['hc_value'])
    gait_events['to_index'] = np.array(gait_events['to_index'])
    gait_events['to_value'] = np.array(gait_events['to_value'])
    gait_events['ms_index'] = 1*mid_swing_index
    gait_events['ms_value'] = 1*mid_swing_value
    
    return gait_events





